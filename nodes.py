"""
ki MiniMax H3 Reference to Video

100% behavior of official MiniMaxH3ReferenceToVideo, plus extra
ref_image_size options that scale by short edge (identical algorithm to "max"):

  match   → area-match the generation canvas (official)
  1.2     → short edge = gen_short
  1.3~3.1 → short edge = gen_short × (ratio/1.2), capped ≤2048
  max     → short edge ≤ 2048 px            (official)

Example at 1.0MP (1376×768, gen_short=768):
  1.2→768 1.3→832 1.4→896 1.5→960 1.6→1024 1.7→1088
  1.8→1152 1.9→1216 2.0→1280 2.1→1344 2.2→1408 2.3→1472
  2.4→1536 2.5→1600 2.6→1664 2.7→1728 2.8→1792 2.9→1856
  3.0→1920 3.1→1984
  max→2048

All modes only downscale (never upscale). Final width/height always
rounded to multiples of 32. Larger short-edge = higher fidelity, slower.
"""

from __future__ import annotations

import math

import torch
import torchaudio

import nodes
import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import node_helpers
from comfy_api.latest import io

# ── reuse official helpers & constants when possible ────────────────────────
try:
    from comfy_extras.nodes_minimax_h3 import (
        CANVAS_MULTIPLE,
        REF_IMAGE_SHORT_EDGE,
        FPS,
        AUDIO_LATENT_FPS,
        adapt_canvas,
        _resize,
        _encode_ref_audio,
        _empty_av_latent,
        align_frame_count,
        video_latent_t,
        temporal_shape,
    )
except ImportError as e:
    raise RuntimeError(
        "ki_MiniMaxH3ReferenceToVideo requires ComfyUI official MiniMax H3 nodes "
        "(comfy_extras.nodes_minimax_h3). Please update ComfyUI to a version that "
        "includes MiniMax H3 support."
    ) from e


def _ref_image_scale(ref_image_size: str, w: int, h: int, width: int, height: int) -> float:
    """Return downscale factor (≤ 1.0). Identical algorithm family to official max.

    match  → area-match generation canvas
    1.2    → short-edge = gen_short          (baseline)
    1.3+   → short-edge = gen_short × (ratio / 1.2)  (align 32, ≤2048)
    max    → short-edge = 2048
    """
    if ref_image_size == "match":
        return min(1.0, math.sqrt((width * height) / (w * h)))

    if ref_image_size == "max":
        target = REF_IMAGE_SHORT_EDGE
    else:
        try:
            ratio = float(ref_image_size)
        except (TypeError, ValueError):
            ratio = 1.2
        gen_short = min(width, height)
        # 1.2 == gen_short; higher ratios scale up from that baseline
        raw = gen_short * (ratio / 1.2)
        target = max(CANVAS_MULTIPLE, round(raw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        # never exceed official max short edge
        target = min(target, REF_IMAGE_SHORT_EDGE)

    return min(1.0, target / min(w, h))


class ki_MiniMaxH3ReferenceToVideo(io.ComfyNode):
    """ref2va with extended short-edge sizing options (ki_ prefix, no conflict)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ki_MiniMaxH3ReferenceToVideo",
            description=(
                "ki version of MiniMax H3 Reference to Video. "
                "Same as official + intermediate short-edge sizes 1.2/1.4/1.6/1.8/2.0. "
                "Use <Picture i> / <Video k> / <Audio j> tags in the prompt."
            ),
            display_name="ki MiniMax H3 Reference to Video",
            category="model/conditioning/minimax",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input(
                    "length",
                    default=124,
                    min=5,
                    max=3600,
                    step=17,
                    tooltip="Frame count at 24 fps (124 ≈ 5s, trained range ~124-362)",
                ),
                io.Combo.Input(
                    "ref_image_size",
                    options=["match", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "2.0", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8", "2.9", "3.0", "3.1", "max"],
                    default="1.6",
                    tooltip=(
                        "Reference image sizing (downscale only, keep aspect).\n"
                        "• match – scale to generation pixel area (fastest)\n"
                        "• 1.2 – short-edge = generation short edge (baseline)\n"
                        "• 1.3~3.1 – short-edge = gen_short × (ratio/1.2), align 32, capped at 2048\n"
                        "  At 1.0MP (768): 768→832→…→1856→1920→1984\n"
                        "• max – official 2048 px short edge\n"
                        "Larger ratio = better identity, slower (tokens every step)."
                    ),
                ),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "ref_image",
                            tooltip="Reference image (never upscaled)",
                        ),
                        prefix="ref_image_",
                        min=0,
                        max=9,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "ref_video",
                            tooltip="Reference video frames at 24 fps (2-15s)",
                        ),
                        prefix="ref_video_",
                        min=0,
                        max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_video_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input(
                            "ref_video_audio",
                            tooltip="Soundtrack paired with the same-index ref_video",
                        ),
                        prefix="ref_video_audio_",
                        min=0,
                        max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input(
                            "ref_audio",
                            tooltip="Standalone reference audio clip",
                        ),
                        prefix="ref_audio_",
                        min=0,
                        max=3,
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        vae,
        audio_vae,
        prompt,
        width,
        height,
        length,
        ref_image_size="1.6",
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
    ) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)

        ref_items = []   # tokenizer presentation order
        ref_blocks = []  # DiT payload, same order

        # ── reference images ────────────────────────────────────────────────
        for img in (ref_images or {}).values():
            if img is None:
                continue
            h, w = img.shape[1], img.shape[2]
            scale = _ref_image_scale(ref_image_size, w, h, width, height)
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img[:1], tw, th, "disabled")
            z = vae.encode(resized)
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({
                "kind": "image",
                "latent_h": th // 16,
                "latent_w": tw // 16,
                "latent": z,
            })

        # ── reference videos (+ optional paired soundtracks) ────────────────
        ref_video_audios = ref_video_audios or {}
        for name, video_frames in (ref_videos or {}).items():
            if video_frames is None:
                continue
            # index-paired soundtrack: ref_video_audio_N belongs to ref_video_N
            soundtrack = ref_video_audios.get(
                "ref_video_audio_" + name.rsplit("_", 1)[-1]
            )
            vh, vw = video_frames.shape[1], video_frames.shape[2]
            cw, ch = adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = _resize(video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            if n < 5:
                raise ValueError(
                    "MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)"
                )
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]
            z = vae.encode(frames)
            audio_latent, ref_audio_t = (None, 0)
            if soundtrack is not None:
                audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
                # soundtrack gets its own <Audio j> label, emitted before <Video k>
                ref_items.append({"type": "audio"})
            # Qwen sees the video at 2 fps with timestamps
            sample_idx = list(range(0, frames.shape[0], FPS // 2))
            qwen_frames = frames[sample_idx]
            ref_items.append({
                "type": "video",
                "data": qwen_frames,
                "timestamps": [i / 2.0 for i in range(len(sample_idx))],
            })
            ref_blocks.append({
                "kind": "video_audio" if ref_audio_t else "video",
                "latent_t": z.shape[2],
                "latent_h": ch // 16,
                "latent_w": cw // 16,
                "ref_audio_t": ref_audio_t,
                "latent": z,
                "audio_latent": audio_latent,
            })

        # ── standalone reference audio ──────────────────────────────────────
        for audio in (ref_audios or {}).values():
            if audio is None:
                continue
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
            ref_items.append({"type": "audio"})
            ref_blocks.append({
                "kind": "audio",
                "ref_audio_t": ref_audio_t,
                "audio_latent": audio_latent,
            })

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
        return io.NodeOutput(cond, latent)


# ── ComfyUI registration (classic + new API) ───────────────────────────────
NODE_CLASS_MAPPINGS = {
    "ki_MiniMaxH3ReferenceToVideo": ki_MiniMaxH3ReferenceToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ki_MiniMaxH3ReferenceToVideo": "ki MiniMax H3 Reference to Video",
}
