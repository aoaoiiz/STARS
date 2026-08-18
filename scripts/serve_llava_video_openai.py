from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
FORMAL_MAX_FRAMES = 16

from creative_video_exp.checkpoint_identity import (
    checkpoint_identity_summary,
    load_checkpoint_manifest,
)
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
    seed: int | None = 42
    stream: bool | None = False


class LlavaVideoService:
    def __init__(self, model_path: str, model_name: str, max_frames: int):
        _validate_formal_max_frames(max_frames)
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

    def generate(
        self,
        request: ChatCompletionRequest,
    ) -> tuple[str, dict[str, Any], int]:
        prompt_text, frames = _extract_prompt_and_frames(request.messages, self.max_frames)
        if not frames:
            raise HTTPException(status_code=400, detail="No image frames were supplied.")
        video = np.stack([np.asarray(frame.convert("RGB")) for frame in frames], axis=0)
        time_instruction = (
            f"The input contains {len(frames)} sparse video frames in chronological order. "
            "Their array positions are not timestamps, output segments, or evidence of the "
            "source-video duration. Follow the requested output-timeline contract and ground "
            "the content only in these visible frames."
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
        repetition_stopper, stopping_criteria = _repetition_stopping_criteria(
            prompt_length=int(input_ids.shape[1])
        )
        with self.lock, torch.inference_mode():
            _set_request_seed(request.seed)
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=[pixel_values],
                modalities=["video"],
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
            )
        includes_prompt = output_ids.shape[1] > input_ids.shape[1] and torch.equal(
            output_ids[:, : input_ids.shape[1]], input_ids
        )
        generated_token_count = int(
            output_ids.shape[1] - input_ids.shape[1]
            if includes_prompt
            else output_ids.shape[1]
        )
        if includes_prompt:
            output_ids = output_ids[:, input_ids.shape[1] :]
        content = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()
        return content, {
            "degenerate_repetition_early_stop": bool(repetition_stopper.triggered),
            "repetition_period_tokens": int(repetition_stopper.period_tokens),
            "repetition_count_threshold": int(repetition_stopper.repeat_count),
            "generated_tokens_at_stop": generated_token_count,
            "max_new_tokens": max_new_tokens,
        }, len(frames)


def _set_request_seed(seed: int | None) -> None:
    value = int(seed if seed is not None else 42)
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _repetition_stopping_criteria(prompt_length: int):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class RepetitiveTailStopper(StoppingCriteria):
        def __init__(self) -> None:
            self.prompt_length = prompt_length
            self.min_generated_tokens = 128
            self.max_period_tokens = 8
            self.repeat_count = 8
            self.triggered = False
            self.period_tokens = 0

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            generated = input_ids[0, self.prompt_length :]
            if int(generated.numel()) < self.min_generated_tokens:
                return False
            for period in range(1, self.max_period_tokens + 1):
                required = period * self.repeat_count
                if int(generated.numel()) < required:
                    continue
                tail = generated[-required:]
                pattern = tail[-period:]
                if torch.equal(tail, pattern.repeat(self.repeat_count)):
                    self.triggered = True
                    self.period_tokens = period
                    return True
            return False

    stopper = RepetitiveTailStopper()
    return stopper, StoppingCriteriaList([stopper])


def _extract_prompt_and_frames(
    messages: list[dict[str, Any]],
    max_frames: int,
) -> tuple[str, list[Image.Image]]:
    _validate_formal_max_frames(max_frames)
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
                    raw_url = image_url.get("url", "")
                else:
                    raw_url = image_url
                url = raw_url if isinstance(raw_url, str) else ""
                if not url.startswith("data:image"):
                    raise HTTPException(
                        status_code=400,
                        detail="STARS accepts only data-image frame inputs.",
                    )
                if len(frames) >= max_frames:
                    raise HTTPException(
                        status_code=400,
                        detail=f"STARS accepts at most {FORMAL_MAX_FRAMES} frames per request.",
                    )
                frames.append(_decode_data_image(url))
    return "\n".join(part for part in text_parts if part), frames


def _validate_formal_max_frames(max_frames: int) -> None:
    if int(max_frames) != FORMAL_MAX_FRAMES:
        raise ValueError(
            f"STARS requires a service frame cap of {FORMAL_MAX_FRAMES}; received {max_frames}."
        )


def _decode_data_image(url: str) -> Image.Image:
    try:
        header, encoded = url.split(",", 1)
        normalized_header = header.lower()
        if not normalized_header.startswith("data:image/") or ";base64" not in normalized_header:
            raise ValueError("Invalid data-image header.")
        data = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(data)) as image:
            return image.convert("RGB")
    except (ValueError, binascii.Error, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid data-image frame.") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="llava_qwen")
    parser.add_argument("--served-model-name", default="LLaVA-Video-7B-Qwen2")
    parser.add_argument(
        "--checkpoint-manifest",
        required=True,
        help="Sidecar produced by scripts/build_checkpoint_manifest.py.",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument(
        "--max-frames",
        type=int,
        choices=(FORMAL_MAX_FRAMES,),
        default=FORMAL_MAX_FRAMES,
    )
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
                    "checkpoint_identity": checkpoint_identity,
                    "max_frames": FORMAL_MAX_FRAMES,
                    "frame_input_policy": "one_to_sixteen_data_images_without_service_resampling",
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
        content, generation_diagnostics, processed_frame_count = service.generate(request)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - started
        usage, requested_frame_count = _request_usage(
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
                processed_frame_count,
                requested_frame_count,
                checkpoint_identity,
                generation_diagnostics,
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
    input_tokens = len(
        tokenizer.encode("\n".join(text_parts), add_special_tokens=True)
    )
    output_tokens = len(tokenizer.encode(output, add_special_tokens=False))
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }, frame_count


def _server_metrics(
    latency: float,
    processed_frame_count: int,
    requested_frame_count: int,
    checkpoint_identity: dict[str, Any],
    generation_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "server_protocol_version": "stars",
        "generation_latency_seconds": round(latency, 6),
        "visual_input_frames": processed_frame_count,
        "requested_visual_input_frames": requested_frame_count,
        "configured_max_frames": FORMAL_MAX_FRAMES,
        "token_accounting": "text tokenizer estimate; visual tokens excluded",
        "checkpoint_identity_sha256": checkpoint_identity["identity_sha256"],
        "checkpoint_model_id": checkpoint_identity["model_id"],
        "checkpoint_revision": checkpoint_identity["revision"],
        **dict(generation_diagnostics or {}),
    }
    if torch.cuda.is_available():
        metrics["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 3)
        metrics["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 1024**2, 3)
    return metrics


if __name__ == "__main__":
    main()
