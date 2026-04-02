import json
import re
import ollama

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

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
        raise ValueError("No JSON object found")
    return json.loads(text[i:j+1])

def parse_one_summary_content(content: str, model="qwen2.5:14b"):
    """
    content: 例如
    "总入量:1720ml其中（静脉用药:1720ml）总出量:570ml其中（尿量:570ml，导尿管:0ml）"
    """
    prompt = f"""
你是 ICU 护理记录的数据结构化助手。
下面是一段“小时小结”的原文，请只做信息抽取，严禁推断、严禁编造。

原文：
{content}

请输出一个 JSON 对象，字段如下（单位都用 ml；没有就填 null）：
{{
  "总入量_ml": null,
  "静脉用药_ml": null,
  "其他入量_ml": null,
  "总出量_ml": null,
  "尿量_ml": null,
  "导尿管_ml": null,
  "其他出量_ml": null,
  "原文": null
}}

规则：
- 只输出 JSON（不要数组，不要解释，不要 markdown）
- 数值只输出数字（例如 1720），不要带 ml
- 如果原文没有明确某字段，填 null
- "原文" 字段原样填入原文（用于追溯）
"""
    resp = ollama.chat(
        model=model,
        messages=[{"role":"user","content":prompt}],
        options={"temperature":0.0, "num_predict":1024}
    )
    obj = extract_json_object(resp["message"]["content"])
    return obj

def batch_parse_summaries(
    summaries_json="summaries.json",
    output_json="summary_structured.json",
    model="qwen2.5:32b"
):
    data = load_json(summaries_json)
    out = []

    for i, item in enumerate(data, 1):
        content = item.get("内容")
        if not content or not str(content).strip():
            out.append({
                "_error": "missing_content",
                "_source_file": item.get("_source_file"),
                "_row_id": item.get("_row_id"),
            })
            continue

        try:
            parsed = parse_one_summary_content(str(content).strip(), model=model)
            # 保留元信息，方便你回填/定位
            parsed["_source_file"] = item.get("_source_file")
            parsed["_row_id"] = item.get("_row_id")
            parsed["小结小时数"] = item.get("小结小时数")
            out.append(parsed)
            print(f"[{i}/{len(data)}] ✅ parsed row_{item.get('_row_id')}")
        except Exception as e:
            out.append({
                "_error": str(e),
                "_source_file": item.get("_source_file"),
                "_row_id": item.get("_row_id"),
                "_raw_content": content
            })
            print(f"[{i}/{len(data)}] ❌ failed row_{item.get('_row_id')}: {e}")

    save_json(out, output_json)
    print(f"\n🎉 done -> {output_json}")
    return out

if __name__ == "__main__":
    batch_parse_summaries(
        summaries_json="/data1/jianf/新提取pdf/hour_summary_run/summaries.json",
        output_json="/data1/jianf/新提取pdf/hour_summary_run/summary_structured.json",
        model="qwen2.5:14b"  # 纯文本解析：用普通 LLM 即可，没必要 VL
    )
