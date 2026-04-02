import os
import json
import re
from pathlib import Path
from collections import defaultdict

def sort_key_from_filename(filename):
    """
    从文件名中提取排序的 Key，确保 Page 2 排在 Page 1 后面。
    假设文件名格式：林昌海_1_1_ICU护理记录单_result.json
    提取出的 PDF序号(1) 和 页码(1) 作为整数排序
    """
    match = re.search(r'_(\d+)_(\d+)_', filename)
    if match:
        pdf_index = int(match.group(1))
        page_num = int(match.group(2))
        return (pdf_index, page_num)
    return (0, 0)

def is_meaningful_row(row):
    """
    🌟 智能滤网：判断这一行是否包含实质性的医疗数据
    只要除了表头和系统标记外，有任何一个医疗字段存在有效值，就保留。
    """
    ignore_keys = ["日期", "时间", "_block_id", "_source_file", "_error"]
    for key, val in row.items():
        if key not in ignore_keys:
            # 如果值不是 None，且转为字符串后不是空白，也不是 "null"
            if val is not None and str(val).strip() != "" and str(val).strip().lower() != "null":
                return True 
    return False

def global_merge_patient_records(json_folder, output_file):
    """
    全局读取、排序、过滤废块、向下填充并合并跨页数据
    """
    json_path = Path(json_folder)
    all_json_files = list(json_path.glob("*_result.json"))
    
    if not all_json_files:
        print("❌ 未找到任何 JSON 文件，请检查路径！")
        return

    # 1. 严格按照 PDF 顺序和页码顺序排序
    all_json_files.sort(key=lambda x: sort_key_from_filename(x.name))
    
    raw_flat_records = []

    # 2. 把所有文件里的行全部按顺序展平到一个大列表里
    for json_file in all_json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for row in data:
                        row['_source_file'] = json_file.name 
                    raw_flat_records.extend(data)
        except Exception as e:
            print(f"⚠️ 读取 {json_file.name} 失败: {e}")

    # 3. 🌟 启动智能滤网，清洗全空废块 (比如全是 null 的 block_00)
    cleaned_records = []
    filtered_count = 0
    for row in raw_flat_records:
        if is_meaningful_row(row):
            cleaned_records.append(row)
        else:
            filtered_count += 1
            # 你可以取消下面这行的注释来查看被扔掉的具体是哪些块
            # print(f"🗑️ 过滤掉全空废块: {row.get('_source_file')} -> {row.get('_block_id')}")

    print(f"🧹 智能滤网工作完毕：清理了 {filtered_count} 个全空废块（如空时间、无实质数据的占位行）。")

    # 4. 跨页全局向下填充 (Global Forward Fill)
    current_date = None
    current_time = None
    
    for row in cleaned_records:
        # 填充日期
        if row.get("日期") and str(row.get("日期")).lower() != "null":
            current_date = row["日期"]
        else:
            row["日期"] = current_date
            
        # 填充时间
        if row.get("时间") and str(row.get("时间")).lower() != "null":
            current_time = row["时间"]
        else:
            row["时间"] = current_time

    # 5. 全局跨页合并 (Global Group By Time)
    final_merged_dict = defaultdict(dict)
    
    for row in cleaned_records:
        date_val = row.get("日期", "未知日期")
        time_val = row.get("时间", "未知时间")
        primary_key = f"{date_val}_{time_val}"
        
        if primary_key not in final_merged_dict:
            final_merged_dict[primary_key] = row.copy()
        else:
            existing_row = final_merged_dict[primary_key]
            for key, val in row.items():
                if key in ["日期", "时间", "_block_id", "_source_file", "_error"]:
                    continue
                    
                if val is not None and str(val).strip() != "" and str(val).strip().lower() != "null":
                    if existing_row.get(key) is None or str(existing_row.get(key)).strip() == "" or str(existing_row.get(key)).lower() == "null":
                        existing_row[key] = val
                    else:
                        # 核心缝合逻辑：多页同时间字段拼接
                        existing_row[key] = f"{existing_row[key]}; {val}"
                        
            existing_row["_block_id"] = f"{existing_row.get('_block_id')}+{row.get('_block_id')}"
            # 用集合去重，避免 _source_file 名字越来越长
            sources = set(existing_row.get("_source_file", "").split(" + "))
            sources.add(row.get("_source_file", ""))
            existing_row["_source_file"] = " + ".join(filter(None, sources))

    # 6. 导出最终的一张大宽表
    final_list = list(final_merged_dict.values())
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 全局合并完成！共生成 {len(final_list)} 条独立时间点的有效医疗记录。")
    print(f"📁 结果已完美保存至: {output_file}")

if __name__ == "__main__":
    # 配置区：存放散乱 JSON 的文件夹
    JSON_FOLDER = "/data1/jianf/final_json_results_record1"  # 替换为你实际存放 _result.json 的目录
    
    # 最终输出的、完全清洗合并好的终极 JSON
    FINAL_OUTPUT = "林昌海_最终医疗记录总表.json" 
    
    global_merge_patient_records(JSON_FOLDER, FINAL_OUTPUT)