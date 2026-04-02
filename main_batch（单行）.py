import os
import json
import subprocess
from pathlib import Path
import ollama
# ==========================================
# 这里保留你之前的 Ollama 识别相关代码
# (包含 PROMPT_L, PROMPT_M, PROMPT_R, extract_single_part 和 process_three_columns_batch)
PROMPT_L =     prompt = """
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
    "给氧方式": "提取并合并所有数值",
}

**严格规则：**
1. 空白单元格必须填 null，严禁编造数据
2. 数值保持原始格式（血压用斜杠，不要拆分）
3. 仅输出纯JSON对象（花括号包裹），不要输出数组（方括号），不要包含 ```json 或任何解释文字
"""

PROMPT_M ="""
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
    "入量_总量": "ml（从早上7:00累计，早上7点后清零），若为空则填 null"
    "出量_总量": "ml，若为空则填 null",
    "出量_尿量": "ml，若为空则填 null",
    "出量_大便_颜色性状": "描述文字（如：黄色/蛋花样），若为空则填 null",
    "出量_其他出量": "描述文字（如：导尿管/0），若为空则填 null",
    "痰_色": "数值，若为空则填 null",
    "痰_量": "数值，若为空则填 null",
    "管路护理": "完整描述（如：气管插管/是///23），若为空或者/则填 null",
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
    "病情观察及处理": "文字描述（可能很长，需原封不动完整提取所有文字并拼接，绝不遗漏）若为空则填 null",
}

**严格规则：**
1. 空白单元格必须填 null，严禁编造数据
2. "病情观察及处理"字段可能包含长文本，需完整提取，不要截断
3. 数值字段仅保留数字，不要添加单位
4. 仅输出纯JSON对象（花括号包裹），不要输出数组，不要包含 ```json 或任何解释文字
"""
    

def extract_single_part(img_path, prompt_text, retries=3):
    """
    单个切片识别，增加重试机制和完善的 JSON 解析
    """
    if not img_path.exists():
        return {"_error": "文件不存在"}

    for attempt in range(retries):
        try:
            response = ollama.chat(
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
            
            # JSON 解析
            try:
                data = json.loads(raw)
                # 兼容性处理：如果模型还是返回了数组，取第一个元素
                if isinstance(data, list):
                    if len(data) > 0:
                        return data[0]
                    else:
                        return {} # 空数组返回空对象
                elif isinstance(data, dict):
                    return data
                else:
                    raise ValueError(f"返回了未知的数据类型: {type(data)}")
                    
            except json.JSONDecodeError:
                print(f"    ⚠️ JSON解析失败 (尝试 {attempt+1}/{retries})")
                if attempt == retries - 1:
                    return {"_error": "JSON解析失败", "_raw": raw[:200]}
                continue

        except Exception as e:
            print(f"    ⚠️ Ollama 调用出错: {e} (尝试 {attempt+1}/{retries})")
            if attempt == retries - 1:
                return {"_error": str(e)}
    
    return {"_error": "重试多次失败"}

def process_three_columns_batch(slice_dir="icu_slices", output_json="final_merged_results.json"):
    slice_path = Path(slice_dir)
    # 查找所有的 L 切片作为基准
    l_files = sorted(slice_path.glob("block_*_L.png"))
    
    if not l_files:
        print(f"❌ 在 {slice_dir} 中未找到切片文件，请检查切割步骤是否成功。")
        return

    results = []
    
    for i, l_img in enumerate(l_files):
        prefix = l_img.stem.replace('_L', '')
        m_img = slice_path / f"{prefix}_M.png"
        r_img = slice_path / f"{prefix}_R.png"
        
        print(f"\n[{i+1}/{len(l_files)}] 正在识别: {prefix}")
        
        # 分布式识别
        data_L = extract_single_part(l_img, PROMPT_L)
        data_M = extract_single_part(m_img, PROMPT_M)
        data_R = extract_single_part(r_img, PROMPT_R)
        
        # 错误检查日志
        if "_error" in data_L: print(f"  ❌ L 列识别失败: {data_L['_error']}")
        if "_error" in data_M: print(f"  ❌ M 列识别失败: {data_M['_error']}")
        if "_error" in data_R: print(f"  ❌ R 列识别失败: {data_R['_error']}")

        # 合并结果 (使用 | 运算符或 update, 这里用解包方式兼容性好)
        merged_row = {**data_L, **data_M, **data_R}
        
        # 清理掉内部错误标记字段，只保留有意义的数据（可选）
        if "_raw" in merged_row: del merged_row["_raw"]
        
        merged_row["_block_id"] = prefix
        results.append(merged_row)

    # 保存最终结果
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n🎉 识别合并完成！已保存至 {output_json}")
    except Exception as e:
        print(f"❌ 保存结果文件失败: {e}")
# ==========================================

def call_paddle_env_to_cut(img_path, output_dir):
    """跨环境调用切割器（调试增强版）"""
    # ⚠️ 确保这里填的是你正确的 Paddle 环境路径
    # 例如：PADDLE_PYTHON_PATH = "/home/jianf/miniconda3/envs/paddle_env/bin/python"
    PADDLE_PYTHON_PATH = "/home/jianf/miniconda3/envs/ocr_legacy/bin/python" 
    
    worker_script = os.path.join(os.path.dirname(__file__), "cutter_worker.py")
    
    # 【修复 1】强制使用绝对路径，防止文件生成到莫名其妙的地方
    abs_img_path = os.path.abspath(img_path)
    abs_out_dir = os.path.abspath(output_dir)
    
    print(f"  ✂️ 正在调用隔离环境执行切割...")
    print(f"     ▶️ 调用的 Python: {PADDLE_PYTHON_PATH}")
    print(f"     ▶️ 切割脚本: {worker_script}")
    print(f"     ▶️ 图片路径: {abs_img_path}")
    
    result = subprocess.run(
        [PADDLE_PYTHON_PATH, worker_script, "--img", abs_img_path, "--out", abs_out_dir],
        capture_output=True,
        text=True
    )
    
    # 【修复 2】强制打印子进程的所有日志，绝不隐藏！
    if result.stdout: 
        print(f"  📝 [子进程正常输出]:\n{result.stdout.strip()}")
    if result.stderr: 
        print(f"  ⚠️ [子进程报错/警告]:\n{result.stderr.strip()}")
    
    if result.returncode != 0:
        raise RuntimeError(f"图片切割子进程崩溃 (返回码 {result.returncode})。")
        
    # 【修复 3】物理检查：如果执行完后文件夹里还是没图，直接抛出异常阻断后续
    if not os.path.exists(abs_out_dir) or not list(Path(abs_out_dir).glob("*.png")):
        raise RuntimeError("切割程序执行完毕，但输出目录中没有生成任何图片！请查看上方的子进程日志。")
    
def batch_process_all_patients(input_folder, temp_slice_dir, final_output_dir):
    """批量处理核心循环"""
    input_path = Path(input_folder)
    # 假设你的记录单都是 png 或 jpg 格式
    image_files = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")))
    
    if not image_files:
        print(f"❌ 在 {input_folder} 中未找到任何图片，请检查路径。")
        return

    print(f"📂 找到 {len(image_files)} 张病历图片，准备开始批量处理...\n" + "="*50)

    # 确保最终输出结果的文件夹存在
    os.makedirs(final_output_dir, exist_ok=True)

    # 遍历处理每一张病人的图片
    for i, img_file in enumerate(image_files, start=1):
        print(f"\n▶️ [第 {i}/{len(image_files)} 张] 正在处理病人记录: {img_file.name}")
        
        # 为每张图片定义一个独立的输出 JSON 文件名
        json_filename = f"{img_file.stem}_result.json"
        json_filepath = os.path.join(final_output_dir, json_filename)
        
        # 如果你想跳过已经处理过的文件，可以取消下面这两行的注释：
        # if os.path.exists(json_filepath):
        #     print("  ⏭️ 已经处理过，跳过...")
        #     continue

        try:
            # 1. 跨环境切图 (自动调用 Env A)
            call_paddle_env_to_cut(str(img_file), output_dir=temp_slice_dir)
            
            # 2. 本地环境识别合并 (在当前 Env B 执行)
            # 注意：这里的 process_three_columns_batch 是上一个回答里给你写的函数
            process_three_columns_batch(slice_dir=temp_slice_dir, output_json=json_filepath)
            
            print(f"✅ {img_file.name} 处理完毕！结果保存在 {json_filepath}")
            
        except Exception as e:
            print(f"❌ {img_file.name} 发生严重错误，已跳过。错误信息: {e}")
            continue

    print("\n🎉🎉 所有病人记录单批量处理完毕！")

if __name__ == "__main__":
    # 配置你的路径
    INPUT_IMAGE_FOLDER = "/data1/jianf/新提取pdf/2026-02-11"  # 存放这20多个病人图片的文件夹
    TEMP_SLICE_FOLDER = "temp_icu_slices"                    # 切割过程中存放碎片的临时文件夹
    FINAL_JSON_FOLDER = "final_json_results"                 # 存放所有病人最终JSON的文件夹
    
    batch_process_all_patients(
        input_folder=INPUT_IMAGE_FOLDER,
        temp_slice_dir=TEMP_SLICE_FOLDER,
        final_output_dir=FINAL_JSON_FOLDER
    )