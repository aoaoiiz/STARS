from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import inspect
import os
import random
import subprocess
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
FORMAL_MAX_FRAMES = 16
VIDEOLLAMA2_BASE_COMMIT = "c0bb03abf6b8a6b9a8dccac006fb4db5d4d9e414"
VIDEOLLAMA2_ENCODER_SOURCE = "videollama2/model/encoder.py"

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
        _validate_formal_max_frames(max_frames)
        if not repo_path:
            raise ValueError("A VideoLLaMA2 Git checkout is required.")
        self.model_path = model_path
        self.max_frames = max_frames
        self.lock = threading.Lock()
        resolved_repo_path = str(Path(repo_path).expanduser().resolve())
        if resolved_repo_path not in sys.path:
            sys.path.insert(0, resolved_repo_path)

        from videollama2 import mm_infer, model_init
        from videollama2.model.encoder import CLIPVisionTower
        from videollama2.utils import disable_torch_init

        self.runtime_identity = _videollama2_runtime_identity(
            resolved_repo_path,
            CLIPVisionTower,
        )
        disable_torch_init()
        self.mm_infer = mm_infer
        self.model, self.processor, self.tokenizer = model_init(
            model_path,
            use_flash_attn=False,
        )
        self.model.eval()
        vision_tower = self.model.get_vision_tower()
        loaded_backend = str(
            getattr(vision_tower.config, "_attn_implementation", "")
        )
        if loaded_backend != "eager":
            raise RuntimeError(
                "VideoLLaMA2 CLIP vision tower must use the eager attention backend."
            )
        self.runtime_identity["loaded_clip_attention_backend"] = loaded_backend

    def generate(self, request: ChatCompletionRequest) -> tuple[str, int, int]:
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
            model_tensor_temporal_frames = _model_tensor_temporal_frame_count(
                video_tensor
            )
            output = self.mm_infer(
                video_tensor,
                instruct,
                model=self.model,
                tokenizer=self.tokenizer,
                modal="video",
                **generation_kwargs,
            )
        return str(output).strip(), len(frames), model_tensor_temporal_frames


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
                raw_url = image_url.get("url", "") if isinstance(image_url, dict) else image_url
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


def _model_tensor_temporal_frame_count(video_tensor: Any) -> int:
    shape = getattr(video_tensor, "shape", None)
    if shape is None or len(shape) < 1:
        raise RuntimeError("VideoLLaMA2 preprocessing returned an invalid video tensor.")
    frame_count = int(shape[0])
    if frame_count < 1:
        raise RuntimeError("VideoLLaMA2 preprocessing returned an empty video tensor.")
    return frame_count


def _videollama2_runtime_identity(
    repo_path: str,
    clip_vision_tower: type,
) -> dict[str, Any]:
    repo_root = Path(repo_path).expanduser().resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(f"VideoLLaMA2 repository not found: {repo_root}")
    source_file_value = inspect.getsourcefile(clip_vision_tower)
    if not source_file_value:
        raise RuntimeError("Unable to identify the VideoLLaMA2 encoder source file.")
    source_file = Path(source_file_value).resolve()
    try:
        source_relative_path = source_file.relative_to(repo_root)
    except ValueError as exc:
        raise RuntimeError(
            "The imported VideoLLaMA2 package does not come from --repo-path."
        ) from exc
    if source_relative_path.as_posix() != VIDEOLLAMA2_ENCODER_SOURCE:
        raise RuntimeError("The imported VideoLLaMA2 encoder source path is unexpected.")
    commit = _git_output(repo_root, "rev-parse", "HEAD").decode("utf-8").strip()
    if commit != VIDEOLLAMA2_BASE_COMMIT:
        raise RuntimeError(
            "VideoLLaMA2 must use base commit "
            f"{VIDEOLLAMA2_BASE_COMMIT}; received {commit}."
        )
    patch_path = PROJECT_ROOT / "patches" / "videollama2_clip_eager.patch"
    if not patch_path.is_file():
        raise FileNotFoundError("Missing patches/videollama2_clip_eager.patch.")
    _git_output(
        repo_root,
        "apply",
        "--check",
        "--reverse",
        "--unidiff-zero",
        "--whitespace=nowarn",
        str(patch_path),
    )
    changed_files = [
        item
        for item in _git_output(
            repo_root,
            "diff",
            "--name-only",
            "HEAD",
            "--",
        )
        .decode("utf-8")
        .splitlines()
        if item
    ]
    if changed_files != [VIDEOLLAMA2_ENCODER_SOURCE]:
        raise RuntimeError(
            "The VideoLLaMA2 checkout must contain only the shipped encoder patch."
        )
    base_source = _git_output(
        repo_root,
        "show",
        f"HEAD:{VIDEOLLAMA2_ENCODER_SOURCE}",
    )
    original_assignment = b'        config._attn_implementation = "flash_attention_2"\n'
    eager_assignment = b'        config._attn_implementation = "eager"\n'
    if base_source.count(original_assignment) != 1:
        raise RuntimeError("The VideoLLaMA2 base encoder source is unexpected.")
    expected_source = base_source.replace(original_assignment, eager_assignment, 1)
    loaded_source = source_file.read_bytes()
    if loaded_source != expected_source:
        raise RuntimeError(
            "The VideoLLaMA2 encoder does not match the exact shipped eager patch."
        )
    init_source = "".join(
        inspect.getsource(clip_vision_tower.__init__).split()
    )
    if (
        'config._attn_implementation="eager"' not in init_source
        or 'config._attn_implementation="flash_attention_2"' in init_source
    ):
        raise RuntimeError(
            "VideoLLaMA2 CLIP vision tower is not configured for eager attention. "
            "Apply patches/videollama2_clip_eager.patch to the official checkout."
        )
    repository_diff = _git_output(
        repo_root,
        "diff",
        "--no-ext-diff",
        "--binary",
        "HEAD",
        "--",
        ".",
    )
    return {
        "repository_commit": commit,
        "repository_diff_sha256": hashlib.sha256(repository_diff).hexdigest(),
        "repository_diff_present": bool(repository_diff),
        "encoder_source": source_relative_path.as_posix(),
        "base_encoder_source_sha256": hashlib.sha256(base_source).hexdigest(),
        "encoder_source_sha256": hashlib.sha256(loaded_source).hexdigest(),
        "expected_encoder_source_sha256": hashlib.sha256(expected_source).hexdigest(),
        "eager_patch_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
        "exact_eager_patch_verified": True,
        "required_clip_attention_backend": "eager",
    }


def _git_output(repo_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"VideoLLaMA2 source provenance could not be resolved: {detail}"
        )
    return completed.stdout


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
    parser.add_argument("--served-model-name", default="VideoLLaMA2-7B-16F")
    parser.add_argument(
        "--checkpoint-manifest",
        required=True,
        help="Sidecar produced by scripts/build_checkpoint_manifest.py.",
    )
    parser.add_argument("--repo-path", default=os.environ.get("VIDEOLLAMA2_REPO", DEFAULT_VIDEOLLAMA2_REPO))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument(
        "--max-frames",
        type=int,
        choices=(FORMAL_MAX_FRAMES,),
        default=FORMAL_MAX_FRAMES,
    )
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()
    if not args.repo_path:
        raise RuntimeError("Set --repo-path or VIDEOLLAMA2_REPO to an official Git checkout.")

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
                    "max_frames": FORMAL_MAX_FRAMES,
                    "frame_input_policy": "one_to_sixteen_data_images_without_service_resampling",
                    "model_tensor_frame_accounting": "reported_after_videollama2_preprocessing",
                    "runtime_identity": service.runtime_identity,
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
        (
            content,
            processed_frame_count,
            model_tensor_temporal_frames,
        ) = service.generate(request)
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
                model_tensor_temporal_frames,
                checkpoint_identity,
                service.runtime_identity,
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
    model_tensor_temporal_frames: int,
    checkpoint_identity: dict[str, Any],
    runtime_identity: dict[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "server_protocol_version": "stars",
        "generation_latency_seconds": round(latency, 6),
        "visual_input_frames": processed_frame_count,
        "requested_visual_input_frames": requested_frame_count,
        "model_tensor_temporal_frames": model_tensor_temporal_frames,
        "configured_max_frames": FORMAL_MAX_FRAMES,
        "token_accounting": "text tokenizer estimate; visual tokens excluded",
        "checkpoint_identity_sha256": checkpoint_identity["identity_sha256"],
        "checkpoint_model_id": checkpoint_identity["model_id"],
        "checkpoint_revision": checkpoint_identity["revision"],
        "generation_runtime_identity": runtime_identity,
    }
    if torch.cuda.is_available():
        metrics["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 3)
        metrics["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 1024**2, 3)
    return metrics


if __name__ == "__main__":
    main()
