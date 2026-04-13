"""
Agent 全局状态定义
------------------
所有 LangGraph 节点共享同一个 ImageProcessingState 对象，
读取或更新自己负责的字段，避免通过函数参数散乱传递。

字段分组：
  输入        — 由调用方在初始化时填入
  中间产物    — 由 file_type_router / cut 节点填入
  提取结果    — 由 extract 节点填入
  合并结果    — 由 merge 节点填入
  Agent 控制  — 控制流 / 错误追踪 / 重试计数
"""
from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional


class ImageProcessingState(TypedDict):
    # ===== 输入 =====
    input_path: str
    """输入路径：可以是单张图片、图片文件夹、PDF 文件，或混合文件夹"""

    model: str
    """推理模型名称，例如 "Qwen/Qwen2.5-VL-72B-Instruct"（vLLM）或 "qwen2.5vl:72b"（Ollama）"""

    llm_backend: str
    """LLM 推理后端："vllm" 或 "ollama"（默认 "vllm"）"""

    llm_base_url: str
    """LLM 服务地址，例如 "http://127.0.0.1:8000/v1"（vLLM）或 "http://127.0.0.1:11434"（Ollama）"""

    output_dir: str
    """最终 JSON 结果保存目录"""

    # ===== 中间产物 =====
    detected_file_type: str
    """
    自动检测到的输入类型：
      "image_record1" — 王主任记录单(一)
      "image_record2" — 王主任记录单(二)
      "image_jin"     — 金主任记录单
      "image_mixed"   — 混合图片文件夹（需先路由分类）
      "pdf"           — PDF 文件（需先转图片）
      "docx"          — DOCX 文件（交由 docx 处理器处理）
      "unknown"       — 无法识别
    """

    image_files: List[str]
    """待处理的图片路径列表（file_type_router 或 pdf_to_image 节点填入）"""

    slice_dirs: Dict[str, str]
    """图片名 → 临时切片目录的映射（cut 节点填入）"""

    # ===== 提取结果 =====
    raw_results: Dict[str, Any]
    """图片名 → 该图片切片识别出的原始行列表（extract 节点填入）"""

    merged_results: List[Dict[str, Any]]
    """全局合并、去重、向下填充后的最终行列表（merge 节点填入）"""

    final_output_path: str
    """合并结果写入的 JSON 文件路径（merge 节点填入）"""

    # ===== 错误追踪 =====
    errors: Dict[str, str]
    """图片名 → 错误描述（任意节点遇到不可恢复的错误时填入）"""

    retry_count: Dict[str, int]
    """图片名 → 已重试次数（qc_check 节点管理）"""

    # ===== Agent 控制 =====
    phase: str
    """
    当前阶段标记，用于条件边路由：
      "detect" → "cut" → "extract" → "merge" → "qc" → "done"
      遇到重试时 phase = "retry"
    """

    pending_images: List[str]
    """尚未成功提取的图片列表（qc_check 节点填入，用于重试）"""
