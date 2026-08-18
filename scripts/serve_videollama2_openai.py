from __future__ import annotations

import argparse
import base64
import io
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from creative_video_exp.checkpoint_identity import (
    checkpoint_identity_summary,
    load_checkpoint_manifest,
)


DEFAULT_VIDEOLLAMA2_REPO = ""


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int | None = 1024
    max_new_tokens: int | None = None
    temperature: float | None = 0.0
    top_p: float | None = None
    seed: int | None = 42
    stream: bool | None = False


class VideoLLaMA2Service:
    def __init__(self, model_path: str, max_frames: int, repo_path: str):
        self.model_path = model_path
        self.max_frames = max_frames
        self.lock = threading.Lock()
        if repo_path and repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        from videollama2 import mm_infer, model_init
        from videollama2.utils import disable_torch_init

        disable_torch_init()
        self.mm_infer = mm_infer
        self.model, self.processor, self.tokenizer = model_init(
            model_path,
            use_flash_attn=False,
        )
        self.model.eval()

    def generate(self, request: ChatCompletionRequest) -> str:
        prompt_text, frames = _extract_prompt_and_frames(request.messages, self.max_frames)
        if not frames:
            raise HTTPException(status_code=400, detail="No image frames were supplied.")

        temperature = float(request.temperature or 0.0)
        generation_kwargs: dict[str, Any] = {
            "do_sample": temperature > 0.0,
            "max_new_tokens": int(request.max_new_tokens or request.max_tokens or 1024),
        }
        if temperature > 0.0:
            generation_kwargs["temperature"] = temperature
        if request.top_p is not None:
            generation_kwargs["top_p"] = float(request.top_p)

        instruct = (
            f"The input contains {len(frames)} sparse video frames in chronological order. "
            "Their array positions are not timestamps, output segments, or evidence of the "
            "source-video duration. Follow the requested output-timeline contract and ground "
            "the content only in these visible frames.\n"
            f"{prompt_text}"
        )
        with self.lock, torch.inference_mode():
            _set_request_seed(request.seed)
            video_tensor = self.processor["video"](frames)
            output = self.mm_infer(
                video_tensor,
                instruct,
                model=self.model,
                tokenizer=self.tokenizer,
                modal="video",
                **generation_kwargs,
            )
        return str(output).strip()


def _set_request_seed(seed: int | None) -> None:
    value = int(seed if seed is not None else 42)
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _extract_prompt_and_frames(
    messages: list[dict[str, Any]],
    max_frames: int,
) -> tuple[str, list[Image.Image]]:
    text_parts: list[str] = []
    frames: list[Image.Image] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            text_parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {})
                url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                if url.startswith("data:image"):
                    frames.append(_decode_data_image(url))
    if max_frames > 0 and len(frames) > max_frames:
        keep = np.linspace(0, len(frames) - 1, max_frames, dtype=int).tolist()
        frames = [frames[idx] for idx in keep]
    return "\n".join(part for part in text_parts if part), frames


def _decode_data_image(url: str) -> Image.Image:
    encoded = url.split(",", 1)[1]
    data = base64.b64decode(encoded)
    return Image.open(io.BytesIO(data)).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default="VideoLLaMA2-7B-16F")
    parser.add_argument(
        "--checkpoint-manifest",
        required=True,
        help="Sidecar produced by scripts/build_checkpoint_manifest.py.",
    )
    parser.add_argument("--repo-path", default=os.environ.get("VIDEOLLAMA2_REPO", DEFAULT_VIDEOLLAMA2_REPO))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()

    checkpoint_manifest = load_checkpoint_manifest(
        args.checkpoint_manifest,
        model_path=args.model_path,
        verify_files=True,
    )
    checkpoint_identity = checkpoint_identity_summary(checkpoint_manifest)
    if checkpoint_identity["model_id"] != args.served_model_name:
        raise RuntimeError(
            "Checkpoint manifest model_id must equal --served-model-name: "
            f"{checkpoint_identity['model_id']!r} != {args.served_model_name!r}."
        )

    app = FastAPI()
    service = VideoLLaMA2Service(args.model_path, args.max_frames, args.repo_path)

    def check_auth(authorization: str | None) -> None:
        if not args.api_key:
            return
        if authorization != f"Bearer {args.api_key}":
            raise HTTPException(status_code=401, detail="Invalid API key")

    @app.get("/v1/models")
    def models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        check_auth(authorization)
        return {
            "object": "list",
            "data": [
                {
                    "id": args.served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                    "checkpoint_identity": checkpoint_identity,
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(
        request: ChatCompletionRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        check_auth(authorization)
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        content = service.generate(request)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - started
        usage, frame_count = _request_usage(
            service.tokenizer, request.messages, content
        )
        return {
            "id": f"chatcmpl-local-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": args.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
            "server_metrics": _server_metrics(
                latency,
                frame_count,
                checkpoint_identity,
            ),
        }

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


def _request_usage(tokenizer, messages: list[dict[str, Any]], output: str) -> tuple[dict[str, int], int]:
    text_parts: list[str] = []
    frame_count = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
                elif item.get("type") == "image_url":
                    frame_count += 1
    try:
        input_tokens = len(tokenizer.encode("\n".join(text_parts), add_special_tokens=True))
        output_tokens = len(tokenizer.encode(output, add_special_tokens=False))
    except Exception:
        input_tokens = 0
        output_tokens = 0
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }, frame_count


def _server_metrics(
    latency: float,
    frame_count: int,
    checkpoint_identity: dict[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "server_protocol_version": "stars",
        "generation_latency_seconds": round(latency, 6),
        "visual_input_frames": frame_count,
        "token_accounting": "text tokenizer estimate; visual tokens excluded",
        "checkpoint_identity_sha256": checkpoint_identity["identity_sha256"],
        "checkpoint_model_id": checkpoint_identity["model_id"],
        "checkpoint_revision": checkpoint_identity["revision"],
    }
    if torch.cuda.is_available():
        metrics["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 3)
        metrics["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 1024**2, 3)
    return metrics


if __name__ == "__main__":
    main()
