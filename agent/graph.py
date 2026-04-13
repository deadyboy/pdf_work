"""
LangGraph Agent 图定义
----------------------
将现有处理流程（切图 → 推理 → 合并 → QC）封装为 LangGraph 节点，
支持：
  - 输入文件类型自动检测（image_record1 / image_record2 / image_jin / pdf）
  - 提取失败后自动重试（最多 MAX_RETRY 次）
  - 多图片并发处理（通过 ThreadPoolExecutor）

依赖：
    pip install langgraph openai json-repair

节点拓扑：
    START
      │
      ▼
  detect_type          ← file_type_router.route_input
      │
      ├── [pdf]  ──▶  convert_pdf  ──┐
      │                              │
      └── [image_*] ──────────────▶  cut_images
                                     │
                                     ▼
                                 extract_slices
                                     │
                                     ▼
                                  merge_results
                                     │
                                     ▼
                                  qc_check
                                  │       │
                              [retry]  [done]
                                  │       │
                              cut_images  END
                                     │
                                    ...
"""
from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from langgraph.graph import StateGraph, START, END
    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False

from agent.state import ImageProcessingState
from agent.file_type_router import route_input, convert_pdf_node
from agent.llm_vllm import make_llm_client

try:
    from global_merger import global_merge_patient_records
    _HAS_GLOBAL_MERGER = True
except ImportError:
    _HAS_GLOBAL_MERGER = False

# 最大重试次数（每张图片）
MAX_RETRY = 2

# 并发线程数（建议 ≤ 可用 GPU 数）
MAX_WORKERS = int(os.environ.get("AGENT_MAX_WORKERS", "4"))

# PaddleOCR 环境 Python 路径（与现有代码保持一致）
PADDLE_PYTHON_PATH = os.environ.get(
    "PADDLE_PYTHON",
    "/home/jianf/miniconda3/envs/ocr_legacy/bin/python",
)


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def _cutter_script(file_type: str) -> str:
    """根据记录单类型返回对应的切割脚本路径。"""
    base = Path(__file__).parent.parent
    mapping = {
        "image_record1": "cutter_worker2（王主任数据一单）.py",
        "image_record2": "cutter_worker2（王主任数据一单）.py",  # 二单暂用同一切割器
        "image_jin": "cutter_worker（金主任数据）.py",
    }
    script_name = mapping.get(file_type, "cutter_worker（金主任数据）.py")
    return str(base / script_name)


def _cut_one_image(img_path: str, output_dir: str, file_type: str) -> None:
    """调用对应切割脚本，将单张图片切成列切片。"""
    script = _cutter_script(file_type)
    result = subprocess.run(
        [PADDLE_PYTHON_PATH, script, "--img", os.path.abspath(img_path),
         "--out", os.path.abspath(output_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"切割失败 [{img_path}]:\n{result.stderr[:500]}"
        )


def _extract_one_image(
    img_name: str,
    slice_dir: str,
    file_type: str,
    llm_client,
) -> List[Dict[str, Any]]:
    """对已切好的切片目录调用 LLM 推理，返回行列表。"""
    from agent.prompts import get_prompts_for_type, SLICE_SUFFIX

    slice_path = Path(slice_dir)
    prompts = get_prompts_for_type(file_type)
    suffix = SLICE_SUFFIX[file_type]

    # 找所有"第一列"切片作为基准
    anchor_suffix = suffix[0]
    anchor_files = sorted(slice_path.glob(f"block_*_{anchor_suffix}.png"))

    results = []
    for anchor in anchor_files:
        prefix = anchor.stem.removesuffix(f"_{anchor_suffix}")
        row: Dict[str, Any] = {"_block_id": prefix}
        for suf, prompt in zip(suffix, prompts):
            img_p = slice_path / f"{prefix}_{suf}.png"
            part_data = llm_client.chat(img_path=img_p, prompt=prompt)
            row.update(part_data)
        results.append(row)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# LangGraph 节点函数
# ──────────────────────────────────────────────────────────────────────────────

def cut_images_node(state: ImageProcessingState) -> dict:
    """
    节点：对 image_files 中的每张图片调用切割脚本，生成 slice_dirs。
    只切 pending_images 中的图片（初次为全部，重试时为失败子集）。
    """
    image_files = state.get("pending_images") or state.get("image_files", [])
    file_type = state.get("detected_file_type", "image_jin")
    output_dir = state.get("output_dir", ".")
    slice_dirs = dict(state.get("slice_dirs") or {})
    errors = dict(state.get("errors") or {})

    for img_path in image_files:
        img_name = Path(img_path).stem
        temp_dir = os.path.join(output_dir, f"_slices_{img_name}")
        try:
            _cut_one_image(img_path, temp_dir, file_type)
            slice_dirs[img_name] = temp_dir
        except Exception as e:
            errors[img_name] = f"cut_error: {e}"

    return {"slice_dirs": slice_dirs, "errors": errors, "phase": "extract"}


def extract_slices_node(state: ImageProcessingState) -> dict:
    """
    节点：对 slice_dirs 中的每个切片目录并发调用 LLM 推理。
    """
    slice_dirs = state.get("slice_dirs") or {}
    file_type = state.get("detected_file_type", "image_jin")
    raw_results = dict(state.get("raw_results") or {})
    errors = dict(state.get("errors") or {})

    llm_client = make_llm_client(
        backend=state.get("llm_backend", "vllm"),
        base_url=state.get("llm_base_url", "http://127.0.0.1:8000/v1"),
        model=state.get("model", "Qwen/Qwen2.5-VL-72B-Instruct"),
    )

    def _extract(img_name: str, slice_dir: str) -> tuple[str, Any]:
        try:
            rows = _extract_one_image(img_name, slice_dir, file_type, llm_client)
            return img_name, rows
        except Exception as e:
            return img_name, {"_error": str(e)}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_extract, name, d): name
            for name, d in slice_dirs.items()
        }
        for future in as_completed(futures):
            img_name, result = future.result()
            if isinstance(result, dict) and "_error" in result:
                errors[img_name] = result["_error"]
            else:
                raw_results[img_name] = result

    return {"raw_results": raw_results, "errors": errors, "phase": "merge"}


def merge_results_node(state: ImageProcessingState) -> dict:
    """
    节点：将所有图片的原始行合并为一张全局有序的宽表，
    执行向下填充（日期/时间）和跨页合并。
    """
    import json

    if not _HAS_GLOBAL_MERGER:
        return {
            "errors": {
                **state.get("errors", {}),
                "__merger": "global_merger 模块未找到，请确认 global_merger.py 在 PYTHONPATH 中",
            },
            "phase": "done",
        }

    raw_results = state.get("raw_results") or {}
    output_dir = state.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)

    # 先把各图片结果写成临时 JSON，再调 global_merger
    tmp_dir = os.path.join(output_dir, "_tmp_json")
    os.makedirs(tmp_dir, exist_ok=True)

    image_files = state.get("image_files", [])
    # 按原始图片顺序写出（确保 global_merger 可以正确排序）
    for img_path in image_files:
        img_name = Path(img_path).stem
        rows = raw_results.get(img_name)
        if rows is not None:
            tmp_file = os.path.join(tmp_dir, f"{img_name}_result.json")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)

    final_output = os.path.join(output_dir, "merged_result.json")
    global_merge_patient_records(tmp_dir, final_output)

    # 清理临时目录
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # 读取合并结果
    with open(final_output, encoding="utf-8") as f:
        merged = json.load(f)

    return {
        "merged_results": merged,
        "final_output_path": final_output,
        "phase": "qc",
    }


def qc_check_node(state: ImageProcessingState) -> dict:
    """
    节点：质量检查。
    检测哪些图片提取失败（errors），决定是否重试。
    """
    errors = state.get("errors") or {}
    retry_count = dict(state.get("retry_count") or {})

    pending = []
    for img_name, err_msg in list(errors.items()):
        current = retry_count.get(img_name, 0)
        if current < MAX_RETRY:
            pending.append(img_name)
            retry_count[img_name] = current + 1

    if pending:
        # 把 img_name 转回完整路径
        image_files = state.get("image_files", [])
        name_to_path = {Path(p).stem: p for p in image_files}
        pending_paths = [name_to_path[n] for n in pending if n in name_to_path]
        return {
            "pending_images": pending_paths,
            "retry_count": retry_count,
            "phase": "retry",
        }

    return {"pending_images": [], "retry_count": retry_count, "phase": "done"}


def cleanup_slices_node(state: ImageProcessingState) -> dict:
    """
    节点（可选）：清理所有临时切片目录，节省磁盘空间。
    """
    for slice_dir in (state.get("slice_dirs") or {}).values():
        shutil.rmtree(slice_dir, ignore_errors=True)
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# 条件路由函数
# ──────────────────────────────────────────────────────────────────────────────

def route_after_detect(state: ImageProcessingState) -> str:
    """detect_type 之后的条件路由。"""
    ft = state.get("detected_file_type", "unknown")
    if ft == "pdf":
        return "convert_pdf"
    if ft.startswith("image"):
        return "cut_images"
    if ft == "docx":
        return "docx_handler"  # 如需集成 docx 处理器，在此扩展
    return END


def route_after_qc(state: ImageProcessingState) -> str:
    """qc_check 之后的条件路由：重试或结束。"""
    if state.get("phase") == "retry" and state.get("pending_images"):
        return "cut_images"
    return END


# ──────────────────────────────────────────────────────────────────────────────
# 图构建
# ──────────────────────────────────────────────────────────────────────────────

def build_agent_graph():
    """
    构建并编译 ICU 记录单处理 Agent 图。

    Returns:
        编译后的 LangGraph CompiledGraph 实例。

    Raises:
        ImportError: 若 langgraph 未安装。
    """
    if not _HAS_LANGGRAPH:
        raise ImportError(
            "build_agent_graph 需要 langgraph：pip install langgraph"
        )

    graph = StateGraph(ImageProcessingState)

    # 注册节点
    graph.add_node("detect_type", route_input)
    graph.add_node("convert_pdf", convert_pdf_node)
    graph.add_node("cut_images", cut_images_node)
    graph.add_node("extract_slices", extract_slices_node)
    graph.add_node("merge_results", merge_results_node)
    graph.add_node("qc_check", qc_check_node)
    graph.add_node("cleanup", cleanup_slices_node)

    # 入口 → 检测文件类型
    graph.add_edge(START, "detect_type")

    # 检测完毕 → 条件分发
    graph.add_conditional_edges(
        "detect_type",
        route_after_detect,
        {
            "convert_pdf": "convert_pdf",
            "cut_images": "cut_images",
            END: END,
        },
    )

    # PDF 转图片完成 → 切图
    graph.add_edge("convert_pdf", "cut_images")

    # 切图 → 推理
    graph.add_edge("cut_images", "extract_slices")

    # 推理 → 合并
    graph.add_edge("extract_slices", "merge_results")

    # 合并 → QC
    graph.add_edge("merge_results", "qc_check")

    # QC → 条件路由（重试 or 结束）
    graph.add_conditional_edges(
        "qc_check",
        route_after_qc,
        {
            "cut_images": "cut_images",
            END: "cleanup",
        },
    )

    # 清理 → 结束
    graph.add_edge("cleanup", END)

    return graph.compile()


# ──────────────────────────────────────────────────────────────────────────────
# 便捷运行入口
# ──────────────────────────────────────────────────────────────────────────────

def run_agent(
    input_path: str,
    output_dir: str = "agent_output",
    model: str = "Qwen/Qwen2.5-VL-72B-Instruct",
    llm_backend: str = "vllm",
    llm_base_url: str = "http://127.0.0.1:8000/v1",
) -> dict:
    """
    运行 ICU 记录单提取 Agent。

    Args:
        input_path:   输入路径（图片 / 图片文件夹 / PDF / DOCX）
        output_dir:   输出目录，最终结果保存为 output_dir/merged_result.json
        model:        推理模型名（vLLM 用 HuggingFace 名，Ollama 用模型标签）
        llm_backend:  "vllm"（推荐）或 "ollama"
        llm_base_url: LLM 服务地址

    Returns:
        最终的 ImageProcessingState dict

    用法示例：
        # 使用 vLLM（推荐）
        result = run_agent(
            input_path="/data/patient_001/",
            output_dir="results/patient_001",
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            llm_backend="vllm",
            llm_base_url="http://127.0.0.1:8000/v1",
        )

        # 使用 Ollama（兼容旧版）
        result = run_agent(
            input_path="/data/patient_001/",
            output_dir="results/patient_001",
            model="qwen2.5vl:72b",
            llm_backend="ollama",
            llm_base_url="http://127.0.0.1:11434",
        )
    """
    app = build_agent_graph()

    initial_state: ImageProcessingState = {
        "input_path": input_path,
        "output_dir": output_dir,
        "model": model,
        "llm_backend": llm_backend,
        "llm_base_url": llm_base_url,
        "detected_file_type": "",
        "image_files": [],
        "slice_dirs": {},
        "raw_results": {},
        "merged_results": [],
        "final_output_path": "",
        "errors": {},
        "retry_count": {},
        "phase": "detect",
        "pending_images": [],
    }

    return app.invoke(initial_state)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python -m agent.graph <输入路径> [输出目录]")
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "agent_output"
    result = run_agent(inp, out)
    print(f"\n✅ 完成！合并结果: {result.get('final_output_path')}")
    print(f"   错误数: {len(result.get('errors', {}))}")
