import os
import json
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from ollama import Client  # 导入 Client 用于指定特定端口的多卡分发
import queue

# ==========================================
# 工具函数区域 (修改为支持 client 指定端口，并隔离GPU)
# ==========================================

# 建立端口到 GPU ID 的映射，防止 PaddleOCR 全部挤在 GPU 0 上
PORT_TO_GPU = {
    11434: "0", 11435: "1", 11436: "2",
    11437: "3", 11438: "4", 11439: "5"
}

# 创建一个全局的安全队列，存放 6 个可用的端口令牌
gpu_port_queue = queue.Queue()
for p in PORT_TO_GPU.keys():
    gpu_port_queue.put(p)

# ==========================================
PROMPT_L = """
你是一个专业的ICU医疗数据录入员。这是一张包含生命体征的局部切片。
该切片已通过预处理确保其内【所有数据行】均属于同一个时间点。

这张图片由两部分组成：
1. **顶部区域**：固定的表头（包含：日期、时间、意识、瞳孔/光反、生命体征、机械通气等列名）
2. **底部区域**：需要提取的患者数据行。
3. **切片特点**：一个字段的内容可能分很多行，其原因既有可能是并列的内容，也有可能是同一内容的不同部分，只是框格里写不下才换行。请务必根据表头的垂直对齐关系，正确理解并提取这些分布在多行的内容。
**核心任务：**
请对切片内的所有数据行进行【垂直扫描】。对于同一个字段，如果内容分布在多行，必须全部提取并使用分号 ";" 进行拼接，严禁遗漏任何一行的数值。
请根据**表头的垂直对齐关系**，精准提取底部的数据，严格按以下JSON格式输出**单个对象**。

**输出格式（注意是对象不是数组）：**
{
    "日期": "格式如 08-29，若为空则填 null",
    "时间": "格式如 08:15，若为空则填 null",
    "意识": "数值（如7、8、9）",
    "瞳孔_左": "格式如 3/2",
    "瞳孔_右": "格式如 3/2",
    "体温": "T (℃)",
    "心率": "HR (次/分)",
    "呼吸": "R (次/分)",
    "血压": "BP (mmHg，格式如 114/81)",
    "血氧": "SpO2 (%)",
    "CVP": "mmHg/cmH2O",
    "人工气道方式": "数值（如2）",
    "插管深度": "cm",
    "呼吸模式": "如 V-A/C",
    "VT": "ml",
    "f": "数值", 
    "FiO2": "数值", 
    "PEEP": "数值", 
    "PC/PS": "数值",
    "给氧方式": "提取并合并所有数值"
}

**严格规则：**
1. 空白单元格必须填 null，严禁编造数据
2. 数值保持原始格式（血压用斜杠，不要拆分）
3. 仅输出纯JSON对象（花括号包裹），不要输出数组（方括号），不要包含 ```json 或任何解释文字
"""

PROMPT_M = """
你是一个专业的ICU医疗数据录入员。这是一张包含出入量、管路记录的局部切片。
该切片已通过预处理确保其内【所有数据行】均属于同一个时间点。
这张图片由两部分组成：
1. **顶部区域**：固定的表头（包含：入量(ml)相关列 - 静脉用药、其他、每时、总量，出量(ml)相关列、痰、护理相关列等）
2. **底部区域**：需要提取的患者数据行（只有一行数据）
3. **切片特点**：一个字段的内容可能分很多行，其原因既有可能是并列的内容，也有可能是同一内容的不同部分，只是框格里写不下才换行。请务必根据表头的垂直对齐关系，正确理解并提取这些分布在多行的内容。
**核心任务：**
请对切片内的所有数据行进行【垂直扫描】。对于同一个字段，如果内容分布在多行（常见于“入量”列），必须全部提取并使用分号 ";" 进行拼接，严禁遗漏任何一行的数值。

**字段提取规则：**
1. **聚合规则**：一个切片只输出一个 JSON 对象。同一列的多行文字/数值需拼接
2. **特殊格式**：
- “管路护理”：需保留完整描述（如：气管插管/是///23），若只有斜杠则视为 null，其中需要额外注意的是：管路护理的内容分布于多行不一定代表不同的护理：
（严格注意此规则）例如第三行是“鼻胃管/是/咖啡色”，第四行是“/50”，此时提取时应该理解为“鼻胃管/是/咖啡色/50”，而不是省略第四行或者将第四行单独作为一段。
3. **入量/出量**：严格按照列名归类数值，注意入量_总量和出量总量靠的很近，注意区分数值所在的位置，不允许漏填以及填错。
请根据**表头的垂直对齐关系**，精准提取底部的数据，严格按以下JSON格式输出**单个对象**。

**输出格式：**
{
    "入量_静脉用药": "ml，若为空则填 null",
    "入量_其他": "ml（非静脉用药，如肌肉注射、皮下注射、口服），若为空则填 null",
    "入量_每时": "ml，若为空则填 null",
    "入量_总量": "ml（从早上7:00累计，早上7点后清零），若为空则填 null",
    "出量_总量": "ml，若为空则填 null",
    "出量_尿量": "ml，若为空则填 null",
    "出量_大便_颜色性状": "描述文字（如：黄色/蛋花样），若为空则填 null",
    "出量_其他出量": "描述文字（如：导尿管/0），若为空则填 null",
    "痰_色": "数值，若为空则填 null",
    "痰_量": "数值，若为空则填 null",
    "管路护理": "完整描述（如：气管插管/是///23），若为空或者/则填 null"
}

**严格规则：**
1. 空白单元格必须填 null，严禁编造数据
2. 数值仅保留数字，不要添加单位（单位已在字段说明中）
3. 仅输出纯JSON对象（花括号包裹），不要输出数组，不要包含 ```json 或任何解释文字
"""
    

PROMPT_R = """
你是一个专业的ICU医疗数据录入员。这是一张包含管路及护理记录的局部切片。
该切片已通过预处理确保其内【所有数据行】均属于同一个时间点。
**核心任务：**
请对切片内的所有数据行进行【垂直扫描】。对于同一个字段，如果内容分布在多行（常见于“病情观察及处理”列），必须全部提取并使用分号 ";" 进行拼接，严禁遗漏任何一行的数值。

**字段提取规则：**
1. **聚合规则**：一个切片只输出一个 JSON 对象。同一列的多行文字/数值需拼接，例如："病情观察": "患者神志清；遵医嘱予拔管"。
2. **特殊格式**：
    - “床头抬高/气切”：只要看到 √、✅等类似手写标记，统一记作“是”（但不要被稀疏像素点干扰）。
这张图片由两部分组成：
1. **顶部区域**：固定的表头（包含：床头抬高、约束部位、体位、气切护理、皮肤护理、护理措施、病情观察及处理等等）
2. **底部区域**：需要提取的患者数据行。
3. **切片特点**：一个字段的内容可能分很多行，其原因既有可能是并列的内容，也有可能是同一内容的不同部分，只是框格里写不下才换行。请务必根据表头的垂直对齐关系，正确理解并提取这些分布在多行的内容。
请根据**表头的垂直对齐关系**，精准提取底部的数据，严格按以下JSON格式输出**单个对象**。

**输出格式：**
{
    "床头抬高30度": "√或者✅或者其他符号（√或者✅✅代表是），原封不动记录保留（如果非空必须识别，记作是），若为空则填 null",
    "约束部位_情况": "数值（如1/1），若为空则填 null",
    "体位": "数值，若为空则填 null",
    "气切护理": "√或者✅或者其他符号（√或者✅✅代表是），若为空则填 null",
    "皮肤护理": "数值（例如1|3），若为空则填 null",
    "护理措施": "数值（例如1|3|5），若为空则填 null",
    "病情观察及处理": "文字描述（可能很长，需原封不动完整提取所有文字并拼接，绝不遗漏）若为空则填 null"
}

**严格规则：**
1. 空白单元格必须填 null，严禁编造数据
2. "病情观察及处理"字段可能包含长文本，需完整提取，不要截断
3. 数值字段仅保留数字，不要添加单位
4. 仅输出纯JSON对象（花括号包裹），不要输出数组，不要包含 ```json 或任何解释文字
"""

# ==========================================
# 工具函数区域 (修改为支持 client 指定端口)
# ==========================================

def call_paddle_env_to_cut(img_path, output_dir, port):
    """跨环境调用切割器（独立环境变量，防止GPU冲突）"""
    PADDLE_PYTHON_PATH = "/home/jianf/miniconda3/envs/ocr_legacy/bin/python" 
    worker_script = os.path.join(os.path.dirname(__file__), "cutter_worker（金主任数据）.py")
    
    abs_img_path = os.path.abspath(img_path)
    abs_out_dir = os.path.abspath(output_dir)
    
    # 隔离环境变量，强行指定 PaddleOCR 使用哪张卡
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = PORT_TO_GPU[port]
    
    result = subprocess.run(
        [PADDLE_PYTHON_PATH, worker_script, "--img", abs_img_path, "--out", abs_out_dir],
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"切割崩溃。\nstderr:\n{result.stderr}")
        
    if not os.path.exists(abs_out_dir) or not list(Path(abs_out_dir).glob("*.png")):
        raise RuntimeError(f"未生成图片！可能未识别到表格。\nstderr:\n{result.stderr}")

def extract_single_part(client, img_path, prompt_text, retries=3):
    """单个切片识别，通过传递进来的 client 使用特定显卡端口"""
    if not img_path.exists():
        return {"_error": "文件不存在"}

    for attempt in range(retries):
        try:
            # ✅ 关键修改：使用 client 代替全局 ollama
            response = client.chat(
                model='qwen2.5vl:72b',
                messages=[{'role': 'user', 'content': prompt_text, 'images': [str(img_path)]}],
                options={"temperature": 0.0, "num_predict": 4096}
            )
            raw = response['message']['content'].strip()
            
            # 清理 Markdown
            if raw.startswith("```json"): raw = raw[7:]
            if raw.startswith("```"): raw = raw[3:]
            if raw.endswith("```"): raw = raw[:-3]
            raw = raw.strip()
            
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    return data[0] if len(data) > 0 else {}
                elif isinstance(data, dict):
                    return data
                else:
                    raise ValueError(f"返回了未知的数据类型: {type(data)}")
            except json.JSONDecodeError:
                if attempt == retries - 1:
                    return {"_error": "JSON解析失败", "_raw": raw[:200]}
                continue

        except Exception as e:
            if attempt == retries - 1:
                return {"_error": str(e)}
    
    return {"_error": "重试多次失败"}

def process_three_columns_batch(client, slice_dir, output_json, port):
    """处理该图片切出来的所有块并合并保存"""
    slice_path = Path(slice_dir)
    l_files = sorted(slice_path.glob("block_*_L.png"))
    
    if not l_files:
        raise RuntimeError(f"在 {slice_dir} 中未找到 L 切片文件。")

    results = []
    
    for i, l_img in enumerate(l_files):
        prefix = l_img.stem.replace('_L', '')
        m_img = slice_path / f"{prefix}_M.png"
        r_img = slice_path / f"{prefix}_R.png"
        
        # print(f"  [端口 {port}] 正在识别行: {prefix} ({i+1}/{len(l_files)})")
        data_L = extract_single_part(client, l_img, PROMPT_L)
        data_M = extract_single_part(client, m_img, PROMPT_M)
        data_R = extract_single_part(client, r_img, PROMPT_R)

        merged_row = {**data_L, **data_M, **data_R}
        if "_raw" in merged_row: del merged_row["_raw"]
        merged_row["_block_id"] = prefix
        results.append(merged_row)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ==========================================
# 并行分发与多线程调度核心
# ==========================================

def process_single_image(img_file, final_output_dir):
    """
    单个线程任务：绝对隔离，动态获取端口
    """
    json_filename = f"{img_file.stem}_result.json"
    json_filepath = os.path.join(final_output_dir, json_filename)
    
    if os.path.exists(json_filepath):
        return f"⏭️ [已跳过] {img_file.name} 已存在"
    
    # 🌟 核心修复 1：用图片名字做临时文件夹，做到物理级别的绝对隔离！
    temp_slice_dir = f"temp_slices_{img_file.stem}"
    
    # 🌟 核心修复 2：从队列里阻塞获取一个空闲的 GPU 端口令牌
    port = gpu_port_queue.get() 
    
    try:
        # 此时打印状态，你就知道谁在用哪张卡了
        print(f"  ▶️ [启动] {img_file.name} -> 分配至 GPU {PORT_TO_GPU[port]} (端口 {port})")
        
        # 切图
        call_paddle_env_to_cut(str(img_file), output_dir=temp_slice_dir, port=port)
        
        # 识别
        client = Client(host=f'http://127.0.0.1:{port}')
        process_three_columns_batch(client, temp_slice_dir, json_filepath, port)
        
        return f"✅ [成功] {img_file.name}"
        
    except Exception as e:
        # 保存事故现场
        if os.path.exists(temp_slice_dir):
            error_img_dir = "failed_debug_images"
            os.makedirs(error_img_dir, exist_ok=True)
            for dbg_img in Path(temp_slice_dir).glob("debug_*.png"):
                shutil.copy(dbg_img, os.path.join(error_img_dir, f"{img_file.stem}_{dbg_img.name}"))
        return f"❌ [失败] {img_file.name} \n详细错误: {e}"
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_slice_dir):
            shutil.rmtree(temp_slice_dir, ignore_errors=True)
            
        # 🌟 核心修复 3：任务完成，将端口令牌归还给队列，供下个图片使用
        gpu_port_queue.put(port)


def batch_process_parallel(input_folder, final_output_dir):
    input_path = Path(input_folder)
    image_files = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")))
    if not image_files: return
    os.makedirs(final_output_dir, exist_ok=True)
    
    max_workers = len(PORT_TO_GPU)
    print(f"📂 找到 {len(image_files)} 张图片，启动 {max_workers} 卡动态负载均衡矩阵！")
    print("="*50)

    # 启动线程池，线程数和卡数保持一致
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for img_file in image_files:
            # 提交任务，不再静态分配端口
            futures.append(executor.submit(process_single_image, img_file, final_output_dir))
            
        for i, future in enumerate(as_completed(futures), start=1):
            msg = future.result()
            print(f"[{i}/{len(image_files)}] {msg}")

    print("\n🎉🎉 所有图片并行处理完毕！")

if __name__ == "__main__":
    # 配置路径
    INPUT_IMAGE_FOLDER = "/data1/jianf/新提取pdf/新数据单病人省立"  # 图片存放地
    FINAL_JSON_FOLDER = "新数据单病人省立"                 # 你之前成功跑出的100张存放地
    
    batch_process_parallel(
        input_folder=INPUT_IMAGE_FOLDER,
        final_output_dir=FINAL_JSON_FOLDER
    )