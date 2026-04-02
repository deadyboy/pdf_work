import os
import re
import json
from pathlib import Path

import cv2
import ollama


# =========================
# 0) 工具
# =========================
ROW_FULL_RE = re.compile(r"row_(\d+)\.png$")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def row_id_from_name(name: str):
    m = ROW_FULL_RE.search(name)
    return int(m.group(1)) if m else None

def extract_json_object(text: str):
    text = (text or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j == -1 or j <= i:
        raise ValueError("No JSON object found in output")
    return json.loads(text[i:j+1])


# =========================
# 1) 行切片（你给的 cut_row 版本）
# =========================
def slice_rows_only(img_path, header_y=526, footer_y=2740, output_dir="row_slices", merge_tol=20):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"图片加载失败: {img_path}")

    h, w, _ = img.shape
    ensure_dir(output_dir)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 15
    )

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 40), 1))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=2)

    contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_ys = sorted([cv2.boundingRect(c)[1] for c in contours])

    valid_ys = [y for y in all_ys if header_y < y < footer_y]
    final_ys = [header_y] + valid_ys + [footer_y]

    merged_ys = []
    for y in final_ys:
        if not merged_ys or y - merged_ys[-1] > merge_tol:
            merged_ys.append(y)

    print(f"📍 区域确认: {header_y} -> {footer_y}")
    print(f"📊 检测到有效行边界: {len(merged_ys)} 条")

    out_files = []
    for i in range(len(merged_ys) - 1):
        y1, y2 = merged_ys[i], merged_ys[i+1]
        row_crop = img[y1:y2, 0:w]
        save_path = os.path.join(output_dir, f"row_{i+1:03d}.png")
        cv2.imwrite(save_path, row_crop)
        out_files.append(save_path)

    print(f"✅ 行切片完成: {len(out_files)} 张 -> {output_dir}")
    return out_files


# =========================
# 2) 小时小结：检测 + 抽取
# =========================
def is_hour_summary_row(img_path, model="qwen2.5vl:72b"):
    """
    用视觉模型做二分类：是否包含“小时小结”
    """
    prompt = "判断图片是否包含“小时小结”这一行（例如“11小时小结：……”）。只回答：是 或 否。"
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [os.path.abspath(img_path)]}],
        options={"temperature": 0.0, "num_predict": 32}
    )
    ans = resp["message"]["content"].strip()
    return ("是" in ans) and ("否" not in ans)

def extract_hour_summary(img_path, model="qwen2.5vl:72b"):
    """
    抽取小时数 + 原文内容
    """
    prompt = """
你是 ICU 护理记录抽取助手。
这张图片是一行“小时小结”（例如“11小时小结：……”）。
请只输出一个 JSON 对象：
{
  "小结小时数": null,
  "内容": null
}
要求：
- 小结小时数为整数（如 11），看不清填 null
- 内容尽量原封不动完整抄写（不要总结，不要改写）
- 只输出 JSON，不要解释，不要 markdown
"""
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [os.path.abspath(img_path)]}],
        options={"temperature": 0.0, "num_predict": 2048}
    )
    obj = extract_json_object(resp["message"]["content"])
    obj["_source_file"] = Path(img_path).name
    obj["_row_id"] = row_id_from_name(Path(img_path).name)
    obj["_image_path"] = str(Path(img_path).resolve())
    obj["类型"] = "小时小结"
    return obj


# =========================
# 3) 主函数：只输出 summaries.json
# =========================
def run_hour_summary_only(
    page_img="/data1/jianf/新提取pdf/data/ICU_Page_1_HighRes.png",
    work_dir="/data1/jianf/新提取pdf/hour_summary_run",
    header_y=526,
    footer_y=2740,
    model="qwen2.5vl:72b",
):
    work_dir = Path(work_dir)
    row_dir = work_dir / "rows_full"
    ensure_dir(str(row_dir))

    # A) 切行
    row_files = slice_rows_only(
        page_img, header_y=header_y, footer_y=footer_y, output_dir=str(row_dir)
    )

    summaries = []
    hits = 0

    # B) 找小时小结行并抽取
    for rf in row_files:
        rid = row_id_from_name(Path(rf).name)
        try:
            if is_hour_summary_row(rf, model=model):
                hits += 1
                s = extract_hour_summary(rf, model=model)
                summaries.append(s)
                print(f"🟦 命中小时小结: row_{rid:03d} -> {rf}")
        except Exception as e:
            print(f"⚠️ row_{rid:03d} 检测/抽取失败: {e}")

    out_json = work_dir / "summaries.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成：命中 {hits} 行小时小结")
    print(f"💾 输出: {out_json}")
    return summaries


if __name__ == "__main__":
    run_hour_summary_only(
        page_img="/data1/jianf/新提取pdf/data/ICU_Page_1_HighRes.png",
        work_dir="/data1/jianf/新提取pdf/hour_summary_run",
        header_y=526,
        footer_y=2740,
        model="qwen2.5vl:72b",
    )
