from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from PIL import Image


def _lazy_import_llava():
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.model.builder import load_pretrained_model

    return {
        "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
        "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
        "conv_templates": conv_templates,
        "tokenizer_image_token": tokenizer_image_token,
        "load_pretrained_model": load_pretrained_model,
    }


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int | None = 1024
    max_new_tokens: int | None = None
    temperature: float | None = 0.0
    top_p: float | None = None
    stream: bool | None = False


class LlavaVideoService:
    def __init__(self, model_path: str, model_name: str, max_frames: int):
        self.model_path = model_path
        self.model_name = model_name
        self.max_frames = max_frames
        self.lock = threading.Lock()
        self.llava = _lazy_import_llava()
        self.tokenizer, self.model, self.image_processor, self.max_length = self.llava[
            "load_pretrained_model"
        ](
            model_path,
            None,
            model_name,
            torch_dtype="bfloat16",
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.model.eval()

    def generate(self, request: ChatCompletionRequest) -> str:
        prompt_text, frames = _extract_prompt_and_frames(request.messages, self.max_frames)
        if not frames:
            raise HTTPException(status_code=400, detail="No image frames were supplied.")
        video = np.stack([np.asarray(frame.convert("RGB")) for frame in frames], axis=0)
        video_time = float(len(frames))
        frame_time = ",".join(f"{idx:.2f}s" for idx in range(len(frames)))
        time_instruction = (
            f"The video lasts for {video_time:.2f} seconds, and {len(frames)} sparse frames "
            f"are sampled from it. These frames are located at {frame_time}. "
            "Please answer the user's request based on these video frames."
        )
        question = (
            self.llava["DEFAULT_IMAGE_TOKEN"]
            + f"\n{time_instruction}\n"
            + prompt_text
        )
        conv = self.llava["conv_templates"]["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = self.llava["tokenizer_image_token"](
            prompt,
            self.tokenizer,
            self.llava["IMAGE_TOKEN_INDEX"],
            return_tensors="pt",
        ).unsqueeze(0).to("cuda")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                151643
                if "qwen" in self.tokenizer.name_or_path.lower()
                else self.tokenizer.eos_token_id
            )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long().to("cuda")
        pixel_values = self.image_processor.preprocess(video, return_tensors="pt")[
            "pixel_values"
        ].to("cuda", dtype=torch.bfloat16)
        max_new_tokens = int(request.max_new_tokens or request.max_tokens or 1024)
        temperature = float(request.temperature or 0.0)
        do_sample = temperature > 0.0
        with self.lock, torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=[pixel_values],
                modalities=["video"],
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                max_new_tokens=max_new_tokens,
            )
        if output_ids.shape[1] > input_ids.shape[1] and torch.equal(
            output_ids[:, : input_ids.shape[1]], input_ids
        ):
            output_ids = output_ids[:, input_ids.shape[1] :]
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


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
                if isinstance(image_url, dict):
                    url = image_url.get("url", "")
                else:
                    url = str(image_url)
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
    parser.add_argument("--model-name", default="llava_qwen")
    parser.add_argument("--served-model-name", default="LLaVA-Video-7B-Qwen2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()

    app = FastAPI()
    service = LlavaVideoService(args.model_path, args.model_name, args.max_frames)

    def check_auth(authorization: str | None) -> None:
        if not args.api_key:
            return
        expected = f"Bearer {args.api_key}"
        if authorization != expected:
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
