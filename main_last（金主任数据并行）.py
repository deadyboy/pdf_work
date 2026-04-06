import os
import json
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from ollama import Client  # 导入 Client 用于指定特定端口的多卡分发
import queue
# 🌟 新增 import（移植自王主任代码）
import re
from json_repair import repair_json

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

# 🌟 新增：normalize_nulls 清洗假 null（移植自王主任代码）
def normalize_nulls(data_dict):
    """将字符串形式的假 null 统一清洗为真实的 None"""
    for key, value in data_dict.items():
        if isinstance(value, str) and value.strip().lower() in ["null", "none", ""]:
            data_dict[key] = None
    return data_dict

# 🌟 新增：患者信息缓存目录和 PROMPT_HEADER（移植自王主任二单）
PATIENT_CACHE_DIR = "patient_base_info"
os.makedirs(PATIENT_CACHE_DIR, exist_ok=True)

PROMPT_HEADER = """
你是一个专业的医疗数据提取员。图片是一张重症护理记录单的最顶部区域。
请提取病人的基础信息，按以下JSON格式输出单个对象：
{
    "姓名": "字符串，若为空则填 null",
    "床号": "字符串，若为空则填 null",
    "住院号": "字符串，若为空则填 null"
}
严格规则：
1. 必须一律使用双引号包裹作为字符串输出。
2. 仅输出纯JSON对象（花括号包裹），不要包含 ```json 等解释文字。
"""

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
1. **强制字符串法则**：无论你提取到的是纯数字（如 14）、带有符号的数字（如 14→12）、还是纯文字，**必须一律使用双引号包裹，作为字符串输出**！
2. **空白处理**：如果红线围成的格子内是空白或无数据，必须直接输出小写的 null（**注意：null 本身不要加双引号**），严禁编造数据。
3. 图片中已经为你绘制了红色的垂直辅助线。红线是严格的列边界，请绝对不要跨越红线读取数据！
4. 数值保持原始格式（血压用斜杠，不要拆分）
5. 仅输出纯JSON对象（花括号包裹），不要输出数组（方括号），不要包含 ```json 或任何解释文字
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
1. **强制字符串法则**：无论你提取到的是纯数字（如 14）、带有符号的数字（如 14→12）、还是纯文字，**必须一律使用双引号包裹，作为字符串输出**！
2. **空白处理**：如果红线围成的格子内是空白或无数据，必须直接输出小写的 null（**注意：null 本身不要加双引号**），严禁编造数据。
3. 图片中已经为你绘制了红色的垂直辅助线。红线是严格的列边界，请绝对不要跨越红线读取数据！
4. 数值仅保留数字，不要添加单位（单位已在字段说明中）
5. 仅输出纯JSON对象（花括号包裹），不要输出数组，不要包含 ```json 或任何解释文字
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
1. **强制字符串法则**：无论你提取到的是纯数字（如 14）、带有符号的数字（如 14→12）、还是纯文字，**必须一律使用双引号包裹，作为字符串输出**！如果内容为空，必须直接输出小写的 null（null 本身不要加双引号）。
2. 图片中已经为你绘制了红色的垂直辅助线。红线是严格的列边界，请绝对不要跨越红线读取数据！
3. "病情观察及处理"字段可能包含长文本，需完整提取，不要截断
4. 数值字段仅保留数字，不要添加单位
5. 仅输出纯JSON对象（花括号包裹），不要输出数组，不要包含 ```json 或任何解释文字
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
    """单个切片识别，通过传递进来的 client 使用特定显卡端口
    🌟 移植自王主任代码：JSON修复 + OCR辅助Prompt + 正则拦截"""
    if not img_path.exists():
        return {"_error": "文件不存在"}

    # 🌟 步骤 1：寻找并读取底层切割器留下的同名 OCR 字典（移植自王主任二单）
    txt_path = img_path.with_suffix('.txt')
    ocr_text = ""
    if txt_path.exists():
        with open(txt_path, 'r', encoding='utf-8') as f:
            ocr_text = f.read().strip()

    # 🌟 步骤 2：动态拼装 Prompt（只有在 OCR 扫到字时才注入，避免空白切片的干扰）
    dynamic_prompt = prompt_text
    if ocr_text:
        dynamic_prompt += f"""\n
=========================================
【OCR 辅助防漏字典】
底层扫描器在当前切片中识别到了以下零散的文本碎片：
[{ocr_text}]

【使用规则】
1. 你依然需要亲自"看图识字"，严格依据图片中表头的垂直对齐关系来提取数据。
2. 上述字典仅为无序的文本碎片，请作为参考，用来核对你是否有漏字、错字（例如极易漏掉的短小字符或标点）。
3. 如果图片中看到的与字典一致，请务必完整提取，绝不遗漏！
=========================================
"""

    for attempt in range(retries):
        try:
            # 🌟 步骤 3：发送动态拼装好的 dynamic_prompt
            response = client.chat(
                model='qwen2.5vl:72b',
                messages=[{'role': 'user', 'content': dynamic_prompt, 'images': [str(img_path)]}],
                options={"temperature": 0.0, "num_predict": 4096}
            )
            raw = response['message']['content']
            
            # 【防御第一关】：用正则强行提取最外层的 {}，砍掉大模型可能附带的废话
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                raw_json_str = match.group(0)
            else:
                raw_json_str = raw

            # 🌟🌟🌟【re.sub 正则拦截】🌟🌟🌟
            # 强制给大模型裸奔的"带符号数值"（如 14→12, 10-20）穿上双引号
            raw_json_str = re.sub(r'(:\s*)([0-9]+(?:→|->|~|-)[0-9]+)(\s*[,}])', r'\1"\2"\3', raw_json_str)

            try:
                # 尝试用标准库解析
                data = json.loads(raw_json_str)
            except json.JSONDecodeError:
                # 🌟🌟🌟【json_repair 修复兜底】🌟🌟🌟
                # 如果标准解析依然失败（比如末尾多了个逗号），让神器自动修补
                data = repair_json(raw_json_str, return_objects=True)
                if not data:
                    raise ValueError("json_repair 也无法修复该字符串")

            # 统一格式化返回
            if isinstance(data, list):
                return data[0] if len(data) > 0 else {}
            elif isinstance(data, dict):
                return data
            else:
                raise ValueError(f"返回了未知的数据类型: {type(data)}")

        except Exception as e:
            if attempt == retries - 1:
                # 彻底失败时，保留原始文本供错题本记录
                return {"_error": f"解析彻底失败", "_raw": raw}
            # 失败则静默重试
            continue

    return {"_error": "重试多次均失败"}

# 🌟 新增：患者信息缓存提取（移植自王主任二单）
def extract_patient_info_once(client, slice_dir, img_name):
    """提取病人基础信息并缓存到本地，已有档案则跳过"""
    # 根据文件命名规范提取姓名，例如 "林昌海_23_7_..." -> "林昌海"
    patient_name = img_name.split('_')[0]
    cache_file = os.path.join(PATIENT_CACHE_DIR, f"{patient_name}.json")
    
    # 如果本地已经有这个人的档案了，直接 return，省下大模型算力！
    if os.path.exists(cache_file):
        return
        
    header_img_path = Path(slice_dir) / "_header_info.png"
    if not header_img_path.exists():
        return
        
    print(f"  💡 [初次建档] 正在为患者【{patient_name}】提取并建立基础信息档案...")
    patient_data = extract_single_part(client, header_img_path, PROMPT_HEADER)
    
    # 写入独立的病人档案 JSON
    if "_error" not in patient_data:
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(patient_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

def process_three_columns_batch(client, slice_dir, output_json, port, img_name):
    """处理该图片切出来的所有块并合并保存
    🌟 新增 img_name 参数、错题本日志、normalize_nulls（移植自王主任代码）"""
    slice_path = Path(slice_dir)
    l_files = sorted(slice_path.glob("block_*_L.png"))
    
    if not l_files:
        raise RuntimeError(f"在 {slice_dir} 中未找到 L 切片文件。")

    results = []
    total_blocks = len(l_files)
    
    # 🌟 新增：在输出目录下准备一个错题本日志文件
    error_log_file = Path(output_json).parent / "llm_json_errors_log.txt"
    
    for i, l_img in enumerate(l_files):
        prefix = l_img.stem.replace('_L', '')
        m_img = slice_path / f"{prefix}_M.png"
        r_img = slice_path / f"{prefix}_R.png"
        
        gpu_id = PORT_TO_GPU[port]
        print(f"  [GPU {gpu_id} | {img_name}] 正在推理: {prefix} ({i+1}/{total_blocks})")
        
        data_L = extract_single_part(client, l_img, PROMPT_L)
        data_M = extract_single_part(client, m_img, PROMPT_M)
        data_R = extract_single_part(client, r_img, PROMPT_R)

        # 🌟 新增：诊断并记录"错题"（移植自王主任一单）
        if "_raw" in data_L or "_raw" in data_M or "_raw" in data_R:
            with open(error_log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"🛑 发现错误: 图片 {img_name} -> 块 {prefix}\n")
                if "_raw" in data_L: f.write(f"【左侧部分返回】:\n{data_L['_raw']}\n\n")
                if "_raw" in data_M: f.write(f"【中间部分返回】:\n{data_M['_raw']}\n\n")
                if "_raw" in data_R: f.write(f"【右侧部分返回】:\n{data_R['_raw']}\n\n")
                f.write(f"{'='*50}\n")

        merged_row = {**data_L, **data_M, **data_R}
        
        # 记录完后，把 _raw 删掉，保持最终 JSON 干净
        if "_raw" in merged_row: del merged_row["_raw"]
        
        merged_row["_block_id"] = prefix
        # 🌟 核心修改：清洗无效的假 null
        merged_row = normalize_nulls(merged_row)
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
        
        # 🌟 新增：尝试建档。如果已经建过，它内部会瞬间 return 跳过（移植自王主任二单）
        extract_patient_info_once(client, temp_slice_dir, img_file.name)
        
        # 🌟 核心修改：将 img_file.name 作为最后一个参数传进去
        process_three_columns_batch(client, temp_slice_dir, json_filepath, port, img_file.name)
        
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