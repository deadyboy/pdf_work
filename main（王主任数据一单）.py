# main_record1.py
import os
import json
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from ollama import Client
import queue

import re
from json_repair import repair_json
PORT_TO_GPU = {
    11434: "0", 11435: "1", 11436: "2",
    11437: "3", 11438: "4", 11439: "5"
}

gpu_port_queue = queue.Queue()
for p in PORT_TO_GPU.keys():
    gpu_port_queue.put(p)

def normalize_nulls(data_dict):
    """将字符串形式的假 null 统一清洗为真实的 None"""
    for key, value in data_dict.items():
        if isinstance(value, str) and value.strip().lower() in ["null", "none", ""]:
            data_dict[key] = None
    return data_dict
# ==========================================
# 适配《记录单一》的 Prompt 区域
# ==========================================

PROMPT_1 = """
你是一个专业的ICU医疗数据录入员。这是一张《ICU重症护理记录单(一)》左侧部分的局部切片。
该切片已通过预处理确保其内【所有数据行】均属于同一个时间点。

这张图片主要包含：日期、时间、以及核心生命体征（体温、心率、呼吸、SPO2、无创血压、有创血压、CVP等）。

**核心任务：**
请对切片内的所有数据行进行【垂直扫描】。严格根据表头的垂直对齐关系，精准提取底部的数据。
注意：血压字段包含多个子列，切勿混淆无创血压和有创血压。按以下JSON格式输出**单个对象**。

**切片特点（极度重要）：**
1. **跨页延续**：这可能是一行从上一页延续下来的记录，所以最左侧的【日期】和【时间】可能是完全空白的。如果是空白，请坚决输出 null，千万不要捏造时间！
2. **多行溢出**：一个字段的内容可能分多行书写。请你务必进行【垂直扫描】，将同一列内分布在多行的内容全部提取并拼接在一起，严禁遗漏！

**输出格式（注意是对象不是数组，顺序必须严格按照这个模版）：**
{
    "日期": "字符串，格式如 09-24，若为空则填 null",
    "时间": "字符串，格式如 14:00，若为空则填 null",
    "体温": "字符串，提取具体数值或者“测不出”等文字，若为空则填 null",
    "心率": "字符串，提取具体数值或者“测不出”等文字，若为空则填 null",
    "呼吸": "字符串，提取具体数值或者“测不出”等文字，若为空则填 null",
    "SPO2": "字符串，提取具体数值或者“测不出”等文字，若为空则填 null",
    "无创血压": "字符串，如 133/99，或者“测不出”等文字，若为空则填 null",
    "有创血压": "字符串，如 128/99，或者“测不出”等文字，若为空则填 null",
    "有创动脉平均压": "字符串，提取具体数值或者“测不出”等文字，若为空则填 null",
    "CVP": "字符串，提取具体数值，若为空则填 null",
    "口服_名称": "字符串，若为空则填 null",
    "口服_量": "字符串，提取具体数值，若为空则填 null",
    "管饲_名称": "字符串，若为空则填 null",
    "管饲_量": "字符串，提取具体数值，若为空则填 null"
}

**严格规则（极其重要）：**
1. **强制字符串法则**：无论你提取到的是纯数字（如 14）、带有符号的数字（如 14→12）、还是纯文字，**必须一律使用双引号包裹，作为字符串输出**！
2. **空白处理**：如果红线围成的格子内是空白或无数据，必须直接输出小写的 null（**注意：null 本身不要加双引号**），严禁编造数据。
3. 图片中已经为你绘制了红色的垂直辅助线。红线是严格的列边界，请绝对不要跨越红线读取数据！
4. 仅输出纯JSON对象（花括号包裹），不要输出数组，不要包含 ```json 或任何解释文字。
"""

PROMPT_2 = """
你是一个专业的ICU医疗数据录入员。这是一张《ICU重症护理记录单(一)》中间部分的局部切片。
该切片已通过预处理确保其内【所有数据行】均属于同一个时间点。

这张图片主要包含患者的液体管理：入量（微泵、静脉输入）和出量，各项评分（RASS评分、疼痛CPOT评分等）。

**核心任务：**
请对切片内的所有数据行进行【垂直扫描】。按以下JSON格式输出**单个对象**。

**切片特点与拼接规则（极度重要）：**
本区域包含药物名称，极大概率会分多行书写！其原因既有可能是并列的内容（如输入了多种不同的液体），也有可能是同一药名太长导致换行。
请你务必进行【垂直扫描】，对于同一个字段（如“静脉输入_名称”）：
1. 如果是同一药名的延续换行，请直接拼接。
2. 如果是不同药名的并列记录（分布在多行），**必须全部提取，并强制使用分号 ";" 将它们连接起来**！
严禁只提取第一行而遗漏下方行！绝不能截断！

**输出格式（注意是对象不是数组，顺序必须严格按照这个模版）：**
{
    "微泵_名称": "字符串，完整提取文字，若有多行独立内容用分号;拼接，若为空则填 null",
    "微泵_量": "字符串，提取具体数值，若有多行用分号;拼接，若为空则填 null",
    "微泵_速度": "字符串，若有多行用分号;拼接，提取具体数值或带有箭头的变化值（如14→12），若为空则填 null",
    "静脉输入_名称": "字符串，完整提取文字，若有多行独立内容用分号;拼接，若为空则填 null",
    "静脉输入_量": "字符串，提取具体数值，若有多行用分号;拼接，若为空则填 null",
    "出量_内容": "字符串，提取文字（如 尿量），若为空则填 null",
    "出量_量": "字符串，提取具体数值，若为空则填 null",
    "出量_颜色": "字符串，提取文字（如 黄色），若为空则填 null",
    "出量_性状": "字符串，提取文字，若为空则填 null",
    "评分_RASS": "字符串，提取具体数值，若为空则填 null",
    "评分_疼痛": "字符串，提取文字+数值（例如CPOT0），若为空则填 null",
    "评分_谵妄": "字符串，提取具体数值，若为空则填 null",
    "评分_压疮": "字符串，提取具体数值，若为空则填 null"
}

**严格规则（极其重要）：**
1. **强制字符串法则**：无论你提取到的是纯数字（如 14）、带有符号的数字（如 14→12）、还是纯文字，**必须一律使用双引号包裹，作为字符串输出**！
2. **空白处理**：如果红线围成的格子内是空白或无数据，必须直接输出小写的 null（**注意：null 本身不要加双引号**），严禁编造数据。
3. 图片中已经为你绘制了红色的垂直辅助线。红线是严格的列边界，请绝对不要跨越红线读取数据！
4. 仅输出纯JSON对象（花括号包裹），不要输出数组，不要包含 ```json 或任何解释文字。
"""

PROMPT_3 = """
你是一个专业的ICU医疗数据录入员。这是一张《ICU重症护理记录单(一)》右侧部分的局部切片。
该切片已通过预处理确保其内【所有数据行】均属于同一个时间点。

这张图片主要包含：极其重要的病情变化与措施（长文本）、以及护士签名。

**核心任务：**
请对切片内的所有数据行进行【垂直扫描】。"病情变化与措施"这一列极大概率包含多行长文本，**必须完整提取每一行文字并进行拼接，绝对不允许省略、截断或遗漏**。按以下JSON格式输出**单个对象**。

**输出格式（注意是对象不是数组，顺序必须严格按照这个模版）：**
{
    "病情变化与措施": "字符串，长文本，必须完整拼接多行描述，若为空则填 null",
    "签名": "字符串，护士姓名，若为空则填 null"
}

**严格规则（极其重要）：**
1. **强制字符串法则**：无论提取到什么内容，**必须一律使用双引号包裹，作为字符串输出**！如果内容为空，必须直接输出小写的 null（null 本身不要加双引号）。
2. "病情变化与措施"包含极高价值的医疗信息，请一字不落地提取图片中该列的全部汉字和符号。
3. 图片中已经为你绘制了红色的垂直辅助线。红线是严格的列边界，请绝对不要跨越红线读取数据！
4. 仅输出纯JSON对象（花括号包裹），不要输出数组，不要包含 ```json 或任何解释文字。
"""

# ==========================================
# 工具函数
# ==========================================

def call_paddle_env_to_cut(img_path, output_dir, port):
    PADDLE_PYTHON_PATH = "/home/jianf/miniconda3/envs/ocr_legacy/bin/python" 
    # 指向新的 Record 1 专用切割器
    worker_script = os.path.join(os.path.dirname(__file__), "cutter_worker2（王主任数据一单）.py")
    
    abs_img_path = os.path.abspath(img_path)
    abs_out_dir = os.path.abspath(output_dir)
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = PORT_TO_GPU[port]
    
    result = subprocess.run(
        [PADDLE_PYTHON_PATH, worker_script, "--img", abs_img_path, "--out", abs_out_dir],
        env=env, capture_output=True, text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"切割崩溃。\nstderr:\n{result.stderr}")


def extract_single_part(client, img_path, prompt_text, retries=3):
    if not img_path.exists():
        return {"_error": "文件不存在"}

    for attempt in range(retries):
        try:
            response = client.chat(
                model='qwen2.5vl:72b',
                messages=[{'role': 'user', 'content': prompt_text, 'images': [str(img_path)]}],
                options={"temperature": 0.0, "num_predict": 4096}
            )
            raw = response['message']['content']
            
            # 【防御第一关】：用正则强行提取最外层的 {}，砍掉大模型可能附带的废话
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                raw_json_str = match.group(0)
            else:
                raw_json_str = raw

            # 🌟🌟🌟【这里就是 re.sub 正则拦截代码】🌟🌟🌟
            # 强制给大模型裸奔的“带符号数值”（如 14→12, 10-20）穿上双引号
            # 即使大模型忘了加引号，这行代码也会在解析前瞬间给它补上
            raw_json_str = re.sub(r'(:\s*)([0-9]+(?:→|->|~|-)[0-9]+)(\s*[,}])', r'\1"\2"\3', raw_json_str)

            try:
                # 尝试用标准库解析
                data = json.loads(raw_json_str)
            except json.JSONDecodeError:
                # 🌟🌟🌟【这里就是 json_repair 修复代码】🌟🌟🌟
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


def process_three_columns_batch(client, slice_dir, output_json, port, img_name):
    slice_path = Path(slice_dir)
    files_1 = sorted(slice_path.glob("block_*_1.png"))
    
    if not files_1:
        raise RuntimeError(f"在 {slice_dir} 中未找到 _1 切片文件。")

    results = []
    total_blocks = len(files_1)
    
    # 🌟 新增：在输出目录下准备一个错题本日志文件
    error_log_file = Path(output_json).parent / "llm_json_errors_log.txt"
    
    for i, img_1 in enumerate(files_1):
        prefix = img_1.stem.removesuffix('_1')
        img_2 = slice_path / f"{prefix}_2.png"
        img_3 = slice_path / f"{prefix}_3.png"
        
        gpu_id = PORT_TO_GPU[port]
        print(f"  [GPU {gpu_id} | {img_name}] 正在推理: {prefix} ({i+1}/{total_blocks})")
        
        data_1 = extract_single_part(client, img_1, PROMPT_1)
        data_2 = extract_single_part(client, img_2, PROMPT_2)
        data_3 = extract_single_part(client, img_3, PROMPT_3)

        # 🌟 新增：诊断并记录“错题”
        if "_raw" in data_1 or "_raw" in data_2 or "_raw" in data_3:
            with open(error_log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"🛑 发现错误: 图片 {img_name} -> 块 {prefix}\n")
                if "_raw" in data_1: f.write(f"【左侧部分返回】:\n{data_1['_raw']}\n\n")
                if "_raw" in data_2: f.write(f"【中间部分返回】:\n{data_2['_raw']}\n\n")
                if "_raw" in data_3: f.write(f"【右侧部分返回】:\n{data_3['_raw']}\n\n")
                f.write(f"{'='*50}\n")

        merged_row = {**data_1, **data_2, **data_3}
        
        # 记录完后，把 _raw 删掉，保持最终 JSON 干净
        if "_raw" in merged_row: del merged_row["_raw"]
        
        merged_row["_block_id"] = prefix
        # 🌟 核心修改 1：清洗无效的假 null
        merged_row = normalize_nulls(merged_row)
        results.append(merged_row)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
def process_single_image(img_file, final_output_dir):
    json_filename = f"{img_file.stem}_result.json"
    json_filepath = os.path.join(final_output_dir, json_filename)
    if os.path.exists(json_filepath): return f"⏭️ [已跳过] {img_file.name} 已存在"
    
    temp_slice_dir = f"temp_slices_{img_file.stem}"
    port = gpu_port_queue.get() 
    
    try:
        print(f"  ▶️ [启动] {img_file.name} -> 分配至 GPU {PORT_TO_GPU[port]} (端口 {port})")
        call_paddle_env_to_cut(str(img_file), output_dir=temp_slice_dir, port=port)
        client = Client(host=f'http://127.0.0.1:{port}')
        process_three_columns_batch(client, temp_slice_dir, json_filepath, port, img_file.name)
        return f"✅ [成功] {img_file.name}"
    except Exception as e:
        return f"❌ [失败] {img_file.name} \n详细错误: {e}"
    finally:
        if os.path.exists(temp_slice_dir): shutil.rmtree(temp_slice_dir, ignore_errors=True)
        gpu_port_queue.put(port)

def batch_process_parallel(input_folder, final_output_dir):
    input_path = Path(input_folder)
    image_files = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")))
    if not image_files: return
    os.makedirs(final_output_dir, exist_ok=True)
    
    max_workers = len(PORT_TO_GPU)
    print(f"📂 启动 {max_workers} 卡并发提取《记录单一》！")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, img, final_output_dir) for img in image_files]
        for future in as_completed(futures): print(future.result())

if __name__ == "__main__":
    # 指向你记录单一的文件夹
    INPUT_IMAGE_FOLDER = "/data1/jianf/新提取pdf/调试"  
    FINAL_JSON_FOLDER = "调试结果"                 
    batch_process_parallel(INPUT_IMAGE_FOLDER, FINAL_JSON_FOLDER)