import os
import json
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from ollama import Client  # 导入 Client 用于指定特定端口的多卡分发
import queue
import re
from json_repair import repair_json
# ==========================================
# 工具函数区域 (GPU 端口映射)
# ==========================================

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
# 核心修改：适配《记录单二》的 Prompt 区域
# ==========================================
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

# 在全局区域建立一个文件夹，专门存放病人的基础档案
PATIENT_CACHE_DIR = "patient_base_info"
os.makedirs(PATIENT_CACHE_DIR, exist_ok=True)

PROMPT_1 = """
你是一个专业的ICU医疗数据录入员。这是一张《ICU重症护理记录单(二)》左侧部分的局部切片。
该切片已通过预处理确保其内【所有数据行】均属于同一个时间点。

这张图片主要包含：日期、时间、意识(GCS)、瞳孔(左/右的大小与光反)、上下肢运动反应、痰液评估、呼吸治疗及吸氧相关基础设置。

**核心任务：**
请对切片内的所有数据行进行【垂直扫描】。严格根据表头的垂直对齐关系，精准提取底部的数据。按以下JSON格式输出**单个对象**。

**切片特点（极度重要）：**
1. **跨页延续**：这可能是一行从上一页延续下来的记录，所以最左侧的【日期】和【时间】可能是完全空白的。如果是空白，请坚决输出 null，千万不要捏造时间！
2. **多行溢出**：一个字段的内容可能分多行书写。请你务必进行【垂直扫描】，将同一列内分布在多行的内容全部提取并用分号 ";" 拼接在一起，严禁遗漏！

**输出格式（注意是对象不是数组，顺序必须严格按照这个模版）：**
{
    "日期": "字符串，格式如 09-24，若为空则填 null",
    "时间": "字符串，格式如 14:00，若为空则填 null",
    "意识": "字符串，如 镇静、清醒等，若为空则填 null",
    "GCS": "字符串，文字/数值，若为空则填 null",
    "瞳孔_左_大小": "字符串，数值如 2.0，若为空则填 null",
    "瞳孔_左_反应": "字符串，如 灵敏、迟钝等，若为空则填 null",
    "瞳孔_右_大小": "字符串，数值如 2.0，若为空则填 null",
    "瞳孔_右_反应": "字符串，如 灵敏、迟钝等，若为空则填 null",
    "运动反应_左上肢": "字符串，提取大写字母（如 B），若为空则填 null",
    "运动反应_右上肢": "字符串，提取大写字母（如 B），若为空则填 null",
    "运动反应_左下肢": "字符串，提取大写字母（如 B），若为空则填 null",
    "运动反应_右下肢": "字符串，提取大写字母（如 B），若为空则填 null",
    "痰_质": "字符串，文字描述（如 稠的），若为空则填 null",
    "痰_量": "字符串，文字描述（如 中量），若为空则填 null",
    "呼吸治疗": "字符串，提取大写字母（如 B），若为空则填 null",
    "吸氧_方式": "字符串，文字描述，若为空则填 null",
    "吸氧_浓度": "字符串，数值，若为空则填 null",
    "吸氧_流量": "字符串，数值，若为空则填 null",
    "气管切开_通畅": "字符串，是/否，若为空则填 null",
    "气管切开_气囊压力": "字符串，数值，若为空则填 null"
}

**严格规则（极其重要）：**
1. **强制字符串法则**：无论你提取到的是纯数字（如 14）、带有符号的数字（如 14→12）、还是纯文字，**必须一律使用双引号包裹，作为字符串输出**！
2. **空白处理**：如果红线围成的格子内是空白或无数据，必须直接输出小写的 null（**注意：null 本身不要加双引号**），严禁编造数据。
3. 图片中已经为你绘制了红色的垂直辅助线。红线是严格的列边界，请绝对不要跨越红线读取数据！
4. 仅输出纯JSON对象（花括号包裹），不要输出数组，不要包含 ```json 或任何解释文字。
"""

PROMPT_2 = """
你是一个专业的ICU医疗数据录入员。这是一张《ICU重症护理记录单(二)》中间部分的局部切片。
该切片已通过预处理确保其内【所有数据行】均属于同一个时间点。

这张图片主要包含：气管套管基础设置，呼吸机模式与参数设置（Mode, PC, PIP, FiO₂, PEEP, RR等）、监测参数等。

**核心任务：**
请对切片内的所有数据行进行【垂直扫描】。严格根据表头的垂直对齐关系，精准提取底部的数据，特别注意区分参数设置下的每一项以及监测参数下的每一项，例如"监测参数_VT_e"的值不允许识别到"监测参数_VT_i"列中。按以下JSON格式输出**单个对象**。

**切片特点（极度重要）：**
有些参数可能分多行书写。请你务必进行【垂直扫描】，将同一列内分布在多行的内容全部提取并用分号 ";" 拼接在一起，严禁遗漏！

**输出格式（注意是对象不是数组，顺序必须严格按照这个模版）：**
{
    "气管插管_通畅": "字符串，是/否，若为空则填 null",
    "气管插管_气囊压力": "字符串，数值，若为空则填 null",
    "气管插管_刻度": "字符串，数值，若为空则填 null",
    "参数设置_Mode": "字符串，如 BIPAP，若为空则填 null",
    "参数设置_PC": "字符串，数值，若为空则填 null",
    "参数设置_PS": "字符串，数值，若为空则填 null",
    "参数设置_FiO2": "字符串，数值，若为空则填 null",
    "参数设置_PEEP": "字符串，数值，若为空则填 null",
    "参数设置_RR": "字符串，数值，若为空则填 null",
    "参数设置_VT": "字符串，数值，若为空则填 null",
    "参数设置_IPAP": "字符串，数值，若为空则填 null",
    "参数设置_EPAP": "字符串，数值，若为空则填 null",
    "参数设置_CPAP": "字符串，数值，若为空则填 null",
    "监测参数_VT_i": "字符串，数值，若为空则填 null",
    "监测参数_VT_e": "字符串，数值，若为空则填 null",
    "监测参数_MV_e": "字符串，字母，若为空则填 null",
    "监测参数_Pp_ea_k": "字符串，字母，若为空则填 null"
}

**严格规则（极其重要）：**
1. **强制字符串法则**：无论你提取到的是纯数字（如 14）、带有符号的数字（如 14→12）、还是纯文字，**必须一律使用双引号包裹，作为字符串输出**！
2. **空白处理**：如果红线围成的格子内是空白或无数据，必须直接输出小写的 null（**注意：null 本身不要加双引号**），严禁编造数据。
3. 图片中已经为你绘制了红色的垂直辅助线。红线是严格的列边界，请绝对不要跨越红线读取数据！
4. 仅输出纯JSON对象（花括号包裹），不要输出数组，不要包含 ```json 或任何解释文字。
"""
    
PROMPT_3 = """
你是一个专业的ICU医疗数据录入员。这是一张《ICU重症护理记录单(二)》右侧部分的局部切片。
该切片已通过预处理确保其内【所有数据行】均属于同一个时间点。

这张图片主要包含：呼吸音、导管评估、皮肤评估等。

**核心任务：**
请对切片内的所有数据行进行【垂直扫描】。严格根据表头的垂直对齐关系，精准提取底部的数据，按以下JSON格式输出**单个对象**。这里的基础护理代号可能由多个字母组成（如 A,J,J），请原封不动提取。

**切片特点（极度重要）：**
导管评估和皮肤评估的【名称】【状态】等极大概率会分多行书写。请你务必进行【垂直扫描】，将同一列内分布在多行的内容全部提取并强制用分号 ";" 完整拼接在一起，其表示同一时间进行的不同操作，严禁截断或遗漏！

**输出格式（注意是对象不是数组，顺序必须严格按照这个模版）：**
{
    "呼吸音_左肺": "字符串，字母（如 F），若为空则填 null",
    "呼吸音_右肺": "字符串，字母（如 F），若为空则填 null",
    "肠鸣音": "字符串，文字描述（如 减弱），若为空则填 null",
    "导管评估_名称": "字符串，完整提取文字描述（如 深静脉置管（）1天），若为空则填 null",
    "导管评估_状态": "字符串，文字（如 通畅），若为空则填 null",
    "导管评估_刻度": "字符串，数值（如12），若为空则填 null",
    "皮肤评估_名称": "字符串，文字描述（如 骶尾部或者同上），若为空则填 null",
    "皮肤评估_状态": "字符串，文字描述（如 骶尾部或者同上），若为空则填 null",
    "皮肤评估_面积": "字符串，文字描述或数值，若为空则填 null",
    "皮肤评估_颜色": "字符串，文字描述（如 色素沉着），若为空则填 null"
}

**严格规则（极其重要）：**
1. **强制字符串法则**：无论你提取到的是纯数字（如 14）、带有符号的数字（如 14→12）、还是纯文字，**必须一律使用双引号包裹，作为字符串输出**！
2. **空白处理**：如果红线围成的格子内是空白或无数据，必须直接输出小写的 null（**注意：null 本身不要加双引号**），严禁编造数据。
3. 图片中已经为你绘制了红色的垂直辅助线。红线是严格的列边界，请绝对不要跨越红线读取数据！
4. 仅输出纯JSON对象（花括号包裹），不要输出数组，不要包含 ```json 或任何解释文字。
"""

PROMPT_4 = """
你是一个专业的ICU医疗数据录入员。这是一张《ICU重症护理记录单(二)》最右侧部分的局部切片。
该切片已通过预处理确保其内【所有数据行】均属于同一个时间点。

这张图片主要包含：血管评估（脉搏）、基础护理（一串字母代号）。

**核心任务：**
请对切片内的所有数据行进行【垂直扫描】。这里的基础护理代号可能由多个字母组成（如 A,J,J），请原封不动提取。

**切片特点（极度重要）：**
基础护理等列可能分多行书写。请你务必进行【垂直扫描】，将同一列内分布在多行的内容全部提取并用分号 ";" 完整拼接在一起，严禁截断或遗漏！

**输出格式（注意是对象不是数组，顺序必须严格按照这个模版）：**
{
    "血管评估_脉搏_上_左": "字符串，字母（如 A），若为空则填 null",
    "血管评估_脉搏_上_右": "字符串，字母（如 B），若为空则填 null",
    "血管评估_脉搏_下_左": "字符串，字母（如 B），若为空则填 null",
    "血管评估_脉搏_下_右": "字符串，字母（如 A），若为空则填 null",
    "基础护理": "字符串，提取所有大写字母组合（如 A,J），不允许越列提取，若为空则填 null"
}


**严格规则：**
1. 基础护理列的字母代表了具体的护理措施，非常重要，绝对不能遗漏或拆分错误。
2. 仅输出纯JSON对象（花括号包裹），不要包含 ```json 或任何解释文字。
3. 图片中已经为你绘制了红色的垂直辅助线。红线是严格的列边界，请绝对不要跨越红线读取数据！如果红线围成的格子内是空白，必须输出 null，严禁编造数据。
4. **强制字符串法则**：无论你提取到的是纯数字（如 14）、带有符号的数字（如 14→12）、还是纯文字，**必须一律使用双引号包裹，作为字符串输出**！
5. **空白处理**：如果红线围成的格子内是空白或无数据，必须直接输出小写的 null（**注意：null 本身不要加双引号**），严禁编造数据。
6.【抗幻觉警告】：图片上清晰写着几个字母就是几个，绝不能凭空捏造、多写或重复输出！不要过度脑补空白区域！
"""

PROMPT_5 = """
你是一个专业的ICU医疗数据录入员。这是一张《ICU重症护理记录单(二)》最右侧部分的局部切片。
该切片已通过预处理确保其内【所有数据行】均属于同一个时间点。

这张图片主要包含：约束方式、肢端血运、签名。

**核心任务：**
请对切片内的所有数据行进行【垂直扫描】。这里的约束方式代号可能由多个字母组成（如 A,J,J），请原封不动提取。

**切片特点（极度重要）：**
约束方式等列可能分多行书写。请你务必进行【垂直扫描】，将同一列内分布在多行的内容全部提取并用分号 ";" 完整拼接在一起，严禁截断或遗漏！

**输出格式（注意是对象不是数组，顺序必须严格按照这个模版）：**
{
    "约束_方式": "字符串，提取所有大写字母组合（如 A,B,C），不允许越列提取，若为空则填 null",
    "肢端血运_颜色": "字符串，文字（如 正常），若为空则填 null",
    "肢端血运_温度": "字符串，文字（如 正常），若为空则填 null",
    "肢端血运_肿胀": "字符串，文字（如 无肿胀），若为空则填 null",
    "签名": "字符串，护士姓名，若有多个用分号拼接，若为空则填 null"
}


**严格规则：**
1. 约束方式列的字母代表了具体的护理措施，非常重要，绝对不能遗漏或拆分错误。
2. 仅输出纯JSON对象（花括号包裹），不要包含 ```json 或任何解释文字。
3. 图片中已经为你绘制了红色的垂直辅助线。红线是严格的列边界，请绝对不要跨越红线读取数据！如果红线围成的格子内是空白，必须输出 null，严禁编造数据。
4. **强制字符串法则**：无论你提取到的是纯数字（如 14）、带有符号的数字（如 14→12）、还是纯文字，**必须一律使用双引号包裹，作为字符串输出**！
5. **空白处理**：如果红线围成的格子内是空白或无数据，必须直接输出小写的 null（**注意：null 本身不要加双引号**），严禁编造数据。
6. 【抗幻觉警告】：图片上清晰写着几个字母就是几个，绝不能凭空捏造、多写或重复输出，不要过度脑补空白区域！请精准识别护士签名，严禁出现文字无限重复的幻觉错误。
"""

# ==========================================
# 跨环境调用切割器
# ==========================================

def call_paddle_env_to_cut(img_path, output_dir, port):
    PADDLE_PYTHON_PATH = "/home/jianf/miniconda3/envs/ocr_legacy/bin/python" 
    worker_script = os.path.join(os.path.dirname(__file__), "cutter_worker2（王主任数据二单）.py")
    
    abs_img_path = os.path.abspath(img_path)
    abs_out_dir = os.path.abspath(output_dir)
    
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

# ==========================================
# Ollama 提取与合并模块
# ==========================================

def extract_single_part(client, img_path, prompt_text, retries=3):
    if not img_path.exists():
        return {"_error": "文件不存在"}

    # 🌟 步骤 1：寻找并读取底层切割器留下的同名 OCR 字典
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
1. 你依然需要亲自“看图识字”，严格依据图片中表头的垂直对齐关系来提取数据。
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
                options={"temperature": 0.0, "num_predict": 8192}
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




def extract_patient_info_once(client, slice_dir, img_name):
    # 根据你的文件命名规范提取姓名，例如 "林昌海_23_7_..." -> "林昌海"
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



def process_five_columns_batch(client, slice_dir, output_json, port, img_name):
    """处理该图片切出来的五个块并合并保存"""
    slice_path = Path(slice_dir)
    files_1 = sorted(slice_path.glob("block_*_1.png"))
    
    if not files_1:
        raise RuntimeError(f"在 {slice_dir} 中未找到 _1 切片文件。")

    results = []
    total_blocks = len(files_1)
    
    for i, img_1 in enumerate(files_1):
        prefix = img_1.stem.removesuffix('_1')
        img_2 = slice_path / f"{prefix}_2.png"
        img_3 = slice_path / f"{prefix}_3.png"
        img_4 = slice_path / f"{prefix}_4.png"
        img_5 = slice_path / f"{prefix}_5.png"  # 🌟 读取第五块图片
        
        gpu_id = PORT_TO_GPU[port]
        print(f"  [GPU {gpu_id} | {img_name}] 正在推理: {prefix} ({i+1}/{total_blocks})")
        
        # 串行发送五块进行提取
        data_1 = extract_single_part(client, img_1, PROMPT_1)
        data_2 = extract_single_part(client, img_2, PROMPT_2)
        data_3 = extract_single_part(client, img_3, PROMPT_3)
        data_4 = extract_single_part(client, img_4, PROMPT_4)
        data_5 = extract_single_part(client, img_5, PROMPT_5) # 🌟 提取第五块

        # 🌟 合并五个 JSON 对象
        merged_row = {**data_1, **data_2, **data_3, **data_4, **data_5}
        if "_raw" in merged_row: del merged_row["_raw"]
        merged_row["_block_id"] = prefix
        results.append(merged_row)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

# ==========================================
# 并行分发调度
# ==========================================

def process_single_image(img_file, final_output_dir):
    """单个线程任务：绝对隔离，动态获取端口"""
    json_filename = f"{img_file.stem}_result.json"
    json_filepath = os.path.join(final_output_dir, json_filename)
    
    if os.path.exists(json_filepath):
        return f"⏭️ [已跳过] {img_file.name} 已存在"
    
    temp_slice_dir = f"temp_slices_{img_file.stem}"
    port = gpu_port_queue.get() 

    
    try:
        print(f"  ▶️ [启动] {img_file.name} -> 分配至 GPU {PORT_TO_GPU[port]} (端口 {port})")
        
        call_paddle_env_to_cut(str(img_file), output_dir=temp_slice_dir, port=port)
        
        client = Client(host=f'http://127.0.0.1:{port}')
                
        # 🌟 新增：尝试建档。如果已经建过，它内部会瞬间 return 跳过
        extract_patient_info_once(client, temp_slice_dir, img_file.name)

        # 🌟 核心修改：将 img_file.name 作为最后一个参数传进去
        process_five_columns_batch(client, temp_slice_dir, json_filepath, port, img_file.name)
        
        return f"✅ [成功] {img_file.name}"
        
    except Exception as e:
        if os.path.exists(temp_slice_dir):
            error_img_dir = "failed_debug_images_record2"
            os.makedirs(error_img_dir, exist_ok=True)
            for dbg_img in Path(temp_slice_dir).glob("debug_*.png"):
                shutil.copy(dbg_img, os.path.join(error_img_dir, f"{img_file.stem}_{dbg_img.name}"))
        return f"❌ [失败] {img_file.name} \n详细错误: {e}"
        
    finally:
        if os.path.exists(temp_slice_dir):
            shutil.rmtree(temp_slice_dir, ignore_errors=True)
        gpu_port_queue.put(port)

def batch_process_parallel(input_folder, final_output_dir):
    input_path = Path(input_folder)
    image_files = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")))
    if not image_files: 
        print("未找到图片文件！")
        return
    os.makedirs(final_output_dir, exist_ok=True)
    
    max_workers = len(PORT_TO_GPU)
    print(f"📂 找到 {len(image_files)} 张《记录单二》图片，启动 {max_workers} 卡并发提取！")
    print("="*50)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for img_file in image_files:
            futures.append(executor.submit(process_single_image, img_file, final_output_dir))
            
        for i, future in enumerate(as_completed(futures), start=1):
            msg = future.result()
            print(f"[{i}/{len(image_files)}] {msg}")

    print("\n🎉🎉 所有《记录单二》图片处理完毕！")

if __name__ == "__main__":
    # 建议将《记录单二》的图片单独放在一个文件夹
    INPUT_IMAGE_FOLDER = "/data1/jianf/新提取pdf/routed_data/data_record2"  
    # 输出结果也独立存放，避免覆盖之前的单一数据
    FINAL_JSON_FOLDER = "/data1/jianf/final_json_results_record2"                 
    
    batch_process_parallel(
        input_folder=INPUT_IMAGE_FOLDER,
        final_output_dir=FINAL_JSON_FOLDER
    )