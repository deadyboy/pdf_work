"""
vLLM 推理客户端
---------------
使用 vLLM 的 OpenAI 兼容 API 提供与现有 Ollama 调用接口一致的封装，
方便从 Ollama 无缝迁移：只需把 OllamaImageClient 换成 VLLMImageClient。

依赖：
    pip install openai

vLLM 启动示例（单卡）：
    vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
        --gpu-memory-utilization 0.95 \
        --max-model-len 8192

vLLM 启动示例（多卡 tensor parallel）：
    vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
        --tensor-parallel-size 4 \
        --gpu-memory-utilization 0.95 \
        --max-model-len 8192
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from openai import OpenAI as _OpenAI
    _HAS_OPENAI = True
except ImportError:
    _OpenAI = None  # type: ignore[assignment,misc]
    _HAS_OPENAI = False

try:
    from json_repair import repair_json
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False


# ---------------------------------------------------------------------------
# 图片编码工具
# ---------------------------------------------------------------------------

def _encode_image_b64(img_path: str | Path) -> str:
    """将本地图片读取为 base64 字符串（vLLM 多模态输入格式）。"""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_media_type(img_path: str | Path) -> str:
    suffix = Path(img_path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }.get(suffix, "image/png")


# ---------------------------------------------------------------------------
# JSON 解析工具（与现有 main_record1.py 保持一致）
# ---------------------------------------------------------------------------

def _parse_json_robust(raw: str) -> Dict[str, Any]:
    """
    多层防御解析 JSON：
    1. 正则提取 {...} 块
    2. 用 re.sub 给裸露数值加引号（如 14→12）
    3. 标准 json.loads
    4. json_repair 兜底
    """
    # 层 1：提取最外层 JSON 对象
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    raw_json = match.group(0) if match else raw

    # 层 2：给带符号数值强制补引号（只匹配箭头符号，不匹配日期短横线）
    raw_json = re.sub(
        r"(:\s*)([0-9]+(?:→|->)[0-9]+)(\s*[,}])",
        r'\1"\2"\3',
        raw_json,
    )

    # 层 3：标准解析
    try:
        data = json.loads(raw_json)
        return data[0] if isinstance(data, list) else data
    except json.JSONDecodeError:
        pass

    # 层 4：json_repair 兜底
    if _HAS_JSON_REPAIR:
        data = repair_json(raw_json, return_objects=True)
        if data:
            return data[0] if isinstance(data, list) else data

    raise ValueError(f"无法解析 JSON，原始内容前 300 字符：{raw[:300]}")


# ---------------------------------------------------------------------------
# vLLM 图片推理客户端
# ---------------------------------------------------------------------------

class VLLMImageClient:
    """
    封装 vLLM OpenAI 兼容 API 的视觉推理客户端。

    与 Ollama 的对应关系：
        Ollama                              VLLMImageClient
        ----------------------------------------
        Client(host=...)                    VLLMImageClient(base_url=..., model=...)
        client.chat(model, messages,        client.chat(img_path, prompt, ...)
                    images=[...])

    用法示例：
        client = VLLMImageClient(
            base_url="http://127.0.0.1:8000/v1",
            model="Qwen/Qwen2.5-VL-72B-Instruct",
        )
        result = client.chat(img_path="/path/to/slice.png", prompt=PROMPT_1)
        print(result)  # {"日期": "09-24", "时间": "14:00", ...}
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        retries: int = 3,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        if _HAS_OPENAI:
            self._client = _OpenAI(base_url=base_url, api_key=api_key)
        else:
            self._client = None  # ImportError raised on first chat() call

    def chat(
        self,
        img_path: str | Path,
        prompt: str,
        *,
        retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        对单张切片图片调用 vLLM 视觉模型，返回解析后的 dict。

        失败时返回 {"_error": "...", "_raw": "..."（可选）}。
        """
        if not Path(img_path).exists():
            return {"_error": f"文件不存在: {img_path}"}

        if self._client is None:
            raise ImportError(
                "VLLMImageClient 需要 openai 包：pip install openai"
            )

        max_attempts = retries if retries is not None else self.retries
        img_b64 = _encode_image_b64(img_path)
        media_type = _image_media_type(img_path)
        raw = ""

        for attempt in range(max_attempts):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{media_type};base64,{img_b64}"
                                    },
                                },
                            ],
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                raw = response.choices[0].message.content or ""
                return _parse_json_robust(raw)

            except Exception as e:
                if attempt == max_attempts - 1:
                    return {
                        "_error": f"推理/解析失败: {e}",
                        "_raw": raw[:500] if raw else "",
                    }

        return {"_error": "重试多次均失败"}


# ---------------------------------------------------------------------------
# Ollama 图片推理客户端（保留兼容性，方便对比/回退）
# ---------------------------------------------------------------------------

class OllamaImageClient:
    """
    基于 ollama.Client 的图片推理客户端，接口与 VLLMImageClient 对齐，
    方便在两者之间切换。

    用法示例：
        client = OllamaImageClient(host="http://127.0.0.1:11434", model="qwen2.5vl:72b")
        result = client.chat(img_path="/path/to/slice.png", prompt=PROMPT_1)
    """

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "qwen2.5vl:72b",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        retries: int = 3,
    ) -> None:
        from ollama import Client  # 按需导入，避免未安装时报错
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self._client = Client(host=host)

    def chat(
        self,
        img_path: str | Path,
        prompt: str,
        *,
        retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not Path(img_path).exists():
            return {"_error": f"文件不存在: {img_path}"}

        max_attempts = retries if retries is not None else self.retries
        raw = ""

        for attempt in range(max_attempts):
            try:
                response = self._client.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [str(img_path)],
                        }
                    ],
                    options={"temperature": self.temperature, "num_predict": self.max_tokens},
                )
                raw = response["message"]["content"] or ""
                return _parse_json_robust(raw)

            except Exception as e:
                if attempt == max_attempts - 1:
                    return {
                        "_error": f"推理/解析失败: {e}",
                        "_raw": raw[:500] if raw else "",
                    }

        return {"_error": "重试多次均失败"}


# ---------------------------------------------------------------------------
# 工厂函数：根据 backend 参数自动选择客户端
# ---------------------------------------------------------------------------

def make_llm_client(
    backend: str = "vllm",
    base_url: str = "http://127.0.0.1:8000/v1",
    model: str = "Qwen/Qwen2.5-VL-72B-Instruct",
    **kwargs,
) -> VLLMImageClient | OllamaImageClient:
    """
    根据 backend 参数返回对应的推理客户端。

    Args:
        backend:  "vllm" 或 "ollama"
        base_url: vLLM 用 "http://host:port/v1"；Ollama 用 "http://host:port"
        model:    模型名
        **kwargs: 传递给客户端构造函数的额外参数

    Returns:
        VLLMImageClient 或 OllamaImageClient 实例

    用法示例：
        # vLLM（推荐）
        client = make_llm_client("vllm", base_url="http://127.0.0.1:8000/v1",
                                 model="Qwen/Qwen2.5-VL-72B-Instruct")

        # Ollama（兼容旧版本）
        client = make_llm_client("ollama", base_url="http://127.0.0.1:11434",
                                 model="qwen2.5vl:72b")
    """
    if backend == "vllm":
        return VLLMImageClient(base_url=base_url, model=model, **kwargs)
    elif backend == "ollama":
        return OllamaImageClient(host=base_url, model=model, **kwargs)
    else:
        raise ValueError(f"不支持的 backend: {backend!r}，请传入 'vllm' 或 'ollama'")
