from __future__ import annotations

import argparse
import base64
import io
import threading
import time
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int | None = 1024
    max_new_tokens: int | None = None
    temperature: float | None = 0.0
    top_p: float | None = None
    stream: bool | None = False


class InternVLService:
    def __init__(self, model_path: str, max_frames: int, image_size: int):
        self.model_path = model_path
        self.max_frames = max_frames
        self.image_size = image_size
        self.lock = threading.Lock()
        self.transform = _build_transform(image_size)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = (
            AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=False,
            )
            .eval()
            .cuda()
        )

    def generate(self, request: ChatCompletionRequest) -> str:
        prompt_text, frames = _extract_prompt_and_frames(request.messages, self.max_frames)
        if not frames:
            raise HTTPException(status_code=400, detail="No image frames were supplied.")

        tensors = [self.transform(frame.convert("RGB")) for frame in frames]
        pixel_values = torch.stack(tensors).to(torch.bfloat16).cuda()
        num_patches_list = [1 for _ in frames]
        frame_prefix = "\n".join(f"Frame {idx + 1}: <image>" for idx in range(len(frames)))
        question = (
            f"{frame_prefix}\n"
            f"The input contains {len(frames)} sparse bin-wise frames sampled from one video. "
            "Answer only from the visual evidence in these frames.\n"
            f"{prompt_text}"
        )
        temperature = float(request.temperature or 0.0)
        generation_config: dict[str, Any] = {
            "max_new_tokens": int(request.max_new_tokens or request.max_tokens or 1024),
            "do_sample": temperature > 0.0,
        }
        if temperature > 0.0:
            generation_config["temperature"] = temperature
        if request.top_p is not None:
            generation_config["top_p"] = float(request.top_p)

        with self.lock, torch.inference_mode():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False,
            )
        if isinstance(response, tuple):
            response = response[0]
        return str(response).strip()


def _build_transform(input_size: int):
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


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
    parser.add_argument("--served-model-name", default="InternVL2.5-8B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()

    app = FastAPI()
    service = InternVLService(args.model_path, args.max_frames, args.image_size)

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
