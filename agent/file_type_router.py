"""
文件类型自动检测与高层路由
--------------------------
给定一个路径（文件 or 文件夹），自动判断输入类型并分配给正确的处理流程。

支持的类型：
  image_record1 — 王主任《记录单(一)》图片（文件或文件夹）
  image_record2 — 王主任《记录单(二)》图片（文件或文件夹）
  image_jin     — 金主任记录单图片（文件或文件夹）
  image_mixed   — 混合图片文件夹（需先由 router.py 分类）
  pdf           — PDF 文件（需先转为图片）
  docx          — DOCX 文件（交给 docx 处理器处理）
  unknown       — 无法识别

检测优先级：
  1. 扩展名匹配（pdf / docx）
  2. 文件夹内容采样（图片 / PDF / DOCX）
  3. OCR 表头关键词匹配（区分记录单一 / 二 / 金主任）
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Tuple

# 图片扩展名集合
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _collect_images(path: str | Path) -> List[Path]:
    """收集路径下（或路径本身）的所有图片文件，按文件名排序。"""
    p = Path(path)
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
        return [p]
    if p.is_dir():
        files = sorted(
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )
        return files
    return []


def _ocr_header_text(img_path: Path, crop_ratio: float = 0.15) -> str:
    """
    裁剪图片顶部 crop_ratio 比例区域，用 PaddleOCR 识别文字。
    若 PaddleOCR 未安装则返回空字符串。
    """
    try:
        import cv2
        from paddleocr import PaddleOCR

        img = cv2.imread(str(img_path))
        if img is None:
            return ""
        H = img.shape[0]
        header = img[0: int(H * crop_ratio), :]
        ocr = PaddleOCR(use_angle_cls=False, lang="ch", show_log=False)
        results = ocr.ocr(header, cls=False)
        text = ""
        if results and results[0]:
            for line in results[0]:
                text += line[1][0]
        return text.replace(" ", "").replace("（", "(").replace("）", ")")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 文件类型检测
# ---------------------------------------------------------------------------

def detect_file_type(input_path: str) -> Tuple[str, List[str]]:
    """
    自动检测输入路径的文件类型，返回 (file_type, image_paths)。

    Args:
        input_path: 图片文件、图片文件夹、PDF 或 DOCX 路径

    Returns:
        file_type:   检测结果字符串（见模块注释）
        image_paths: 图片路径列表（对 docx/unknown 返回空列表）

    用法示例：
        file_type, images = detect_file_type("/data/patient_records/")
        # file_type = "image_mixed"
        # images    = ["/data/patient_records/page_001.png", ...]
    """
    p = Path(input_path)

    # ── 0. 扩展名快速判断（无需文件系统访问）──────────────────────────────────
    ext = p.suffix.lower()
    if ext == ".pdf":
        return "pdf", []
    if ext in (".docx", ".doc"):
        return "docx", []

    # ── 1. 单图片文件 ────────────────────────────────────────────────────────
    if p.is_file():
        if ext in IMAGE_EXTS:
            file_type = _classify_single_image(p)
            return file_type, [str(p)]
        return "unknown", []

    # ── 2. 文件夹：按内容采样判断 ────────────────────────────────────────────
    if p.is_dir():
        all_files = list(p.iterdir())
        ext_counts: dict[str, int] = {}
        for f in all_files:
            if f.is_file():
                e = f.suffix.lower()
                ext_counts[e] = ext_counts.get(e, 0) + 1

        has_images = any(e in IMAGE_EXTS for e in ext_counts)
        has_pdf = ".pdf" in ext_counts
        has_docx = ".docx" in ext_counts or ".doc" in ext_counts

        if has_docx and not has_images and not has_pdf:
            return "docx", []

        if has_pdf and not has_images:
            return "pdf", []

        if has_images:
            images = _collect_images(p)
            if not images:
                return "unknown", []
            file_type = _classify_folder(images)
            return file_type, [str(f) for f in images]

    return "unknown", []


def _classify_single_image(img_path: Path) -> str:
    """对单张图片做表头 OCR，判断记录单类型。"""
    text = _ocr_header_text(img_path)
    return _classify_by_header_text(text)


def _classify_folder(images: List[Path]) -> str:
    """
    对文件夹内前 3 张图片采样做 OCR，多数投票决定类型。
    如果票数相同或有多种类型，返回 "image_mixed"。
    """
    sample = images[:3]
    votes: dict[str, int] = {}
    for img in sample:
        t = _classify_single_image(img)
        votes[t] = votes.get(t, 0) + 1

    if len(votes) == 1:
        return next(iter(votes))  # 全部一致

    # 多种类型混合
    return "image_mixed"


def _classify_by_header_text(text: str) -> str:
    """根据 OCR 识别出的表头文字返回记录单类型。"""
    # 王主任记录单(一)
    if "单(一)" in text or "单(1)" in text or "入量" in text or "出量" in text:
        return "image_record1"

    # 王主任记录单(二)
    if "单(二)" in text or "单(2)" in text or "神经系统" in text or "导管评估" in text:
        return "image_record2"

    # 金主任记录单（特征：含有"意识"和"机械通气"等列名，或含"生命体征"）
    if "生命体征" in text or ("意识" in text and "机械通气" in text):
        return "image_jin"

    # 无法识别
    return "unknown"


# ---------------------------------------------------------------------------
# 文件路由入口（供 Agent graph 节点调用）
# ---------------------------------------------------------------------------

def route_input(state: dict) -> dict:
    """
    LangGraph 节点函数：检测输入类型，写入 state。

    更新字段：
      state["detected_file_type"]
      state["image_files"]
      state["phase"]

    用法（在 graph.py 里注册为节点）：
        graph.add_node("detect_file_type", route_input)
    """
    input_path = state.get("input_path", "")
    file_type, image_paths = detect_file_type(input_path)

    return {
        "detected_file_type": file_type,
        "image_files": image_paths,
        "phase": "cut" if file_type.startswith("image") else file_type,
    }


# ---------------------------------------------------------------------------
# PDF → 图片转换（可选，依赖 pdf2image）
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 200) -> List[str]:
    """
    将 PDF 每页渲染为 PNG 图片，返回图片路径列表。

    依赖：pip install pdf2image（需系统安装 poppler）
    """
    try:
        from pdf2image import convert_from_path
    except ImportError as e:
        raise ImportError(
            "pdf_to_images 需要 pdf2image：pip install pdf2image"
        ) from e

    os.makedirs(output_dir, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=dpi)
    paths = []
    stem = Path(pdf_path).stem
    for i, img in enumerate(images):
        out_path = os.path.join(output_dir, f"{stem}_page_{i + 1:04d}.png")
        img.save(out_path, "PNG")
        paths.append(out_path)
    return paths


def convert_pdf_node(state: dict) -> dict:
    """
    LangGraph 节点：将 PDF 转为图片，更新 state["image_files"]。
    仅当 detected_file_type == "pdf" 时调用。
    """
    import tempfile

    input_path = state.get("input_path", "")
    output_dir = os.path.join(state.get("output_dir", "."), "_pdf_pages")
    try:
        image_paths = pdf_to_images(input_path, output_dir)
        # PDF 转好后，再判断记录单类型
        images = [Path(p) for p in image_paths]
        file_type = _classify_folder(images) if images else "unknown"
        return {
            "image_files": image_paths,
            "detected_file_type": file_type,
            "phase": "cut",
        }
    except Exception as e:
        return {
            "errors": {**state.get("errors", {}), "__pdf_convert": str(e)},
            "phase": "done",
        }
