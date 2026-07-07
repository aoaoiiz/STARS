from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import threading
import time
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel


DEFAULT_VIDEOLLAMA2_REPO = ""


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int | None = 1024
    max_new_tokens: int | None = None
    temperature: float | None = 0.0
    top_p: float | None = None
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
            f"The input contains {len(frames)} sparse bin-wise frames sampled from one video. "
            "Answer only from the visual evidence in these frames.\n"
            f"{prompt_text}"
        )
        with self.lock, torch.inference_mode():
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
    parser.add_argument("--repo-path", default=os.environ.get("VIDEOLLAMA2_REPO", DEFAULT_VIDEOLLAMA2_REPO))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()

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
        content = service.generate(request)
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
        }

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
