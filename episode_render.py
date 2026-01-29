#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import os
import platform
import random
import re
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# --- Pillow/MoviePy compatibility patch (Pillow 10+ removed Image.ANTIALIAS) ---
try:
    from PIL import Image
    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
except Exception:
    Image = None  # type: ignore

import numpy as np

try:
    # moviepy 2.x
    from moviepy import AudioFileClip, VideoFileClip, ImageClip, concatenate_videoclips
except Exception:
    # moviepy 1.x
    from moviepy.editor import AudioFileClip, VideoFileClip, ImageClip, concatenate_videoclips

# Optional: gaussian blur (varia entre versões)
try:
    from moviepy.video.fx.gaussian_blur import gaussian_blur as _gaussian_blur_fx
except Exception:
    try:
        from moviepy.video.fx.all import gaussian_blur as _gaussian_blur_fx
    except Exception:
        _gaussian_blur_fx = None

from episode_plan import Scene

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".m4v", ".webm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

SCENE_CUT_LEAD = 0.15  # segundos: adianta a próxima cena (0.10–0.20 é o range bom)

# --------------------------
# MoviePy compat
# --------------------------
def _subclip_compat(clip: VideoFileClip, start: float, end: float) -> VideoFileClip:
    if hasattr(clip, "subclip"):
        return clip.subclip(start, end)
    if hasattr(clip, "subclipped"):
        return clip.subclipped(start, end)
    raise RuntimeError("MoviePy: não encontrei subclip/subclipped nesta versão.")


def _resize_compat(clip, *, width: Optional[int] = None, height: Optional[int] = None):
    """
    Compat para VideoFileClip e ImageClip.
    """
    if hasattr(clip, "resize"):
        return clip.resize(width=width, height=height)
    if hasattr(clip, "resized"):
        return clip.resized(width=width, height=height)
    raise RuntimeError("MoviePy: não encontrei resize/resized nesta versão.")


def crop_compat(clip, x1: int, y1: int, x2: int, y2: int):
    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(clip.w, x2))
    y2 = int(min(clip.h, y2))

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"crop inválido: ({x1},{y1})-({x2},{y2}) para clip {clip.w}x{clip.h}")

    if hasattr(clip, "crop"):
        return clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)

    def _slice(frame):
        return frame[y1:y2, x1:x2]

    if hasattr(clip, "transform"):
        try:
            newclip = clip.transform(lambda gf, t: _slice(gf(t)), apply_to=["mask"])
        except TypeError:
            newclip = clip.transform(lambda frame: _slice(frame))
        newclip.size = (x2 - x1, y2 - y1)
        return newclip

    if hasattr(clip, "fl"):
        def _crop_frame(get_frame, t):
            return _slice(get_frame(t))
        newclip = clip.fl(_crop_frame, apply_to=["mask"])
        newclip.size = (x2 - x1, y2 - y1)
        return newclip

    raise RuntimeError("Este MoviePy não suporta crop via crop/transform/fl.")


def loop_or_trim(clip: VideoFileClip, target_dur: float) -> VideoFileClip:
    if target_dur <= 0:
        return _subclip_compat(clip, 0, 0.01)

    if clip.duration >= target_dur:
        return _subclip_compat(clip, 0, target_dur)

    remaining = target_dur
    parts: List[VideoFileClip] = []
    while remaining > 0:
        take = min(clip.duration, remaining)
        parts.append(_subclip_compat(clip, 0, take))
        remaining -= take

    return concatenate_videoclips(parts, method="chain")


# --------------------------
# Helpers (math / deterministic RNG)
# --------------------------
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _smoothstep01(t: float) -> float:
    # 3t^2 - 2t^3
    t = _clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _seed_from_text(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _rng_from_key(key: str) -> random.Random:
    return random.Random(_seed_from_text(key))


def _apply_multiplier_to_zoom(zoom_value: float, multiplier: float) -> float:
    """
    zoom_value é algo como 1.06.
    multiplier escala apenas a parte acima de 1.0:
      eff = 1.0 + (zoom_value - 1.0) * multiplier
    """
    multiplier = float(multiplier)
    z = float(zoom_value)
    return max(1.0, 1.0 + (z - 1.0) * multiplier)


def _quantize_px(v: float, step: int) -> int:
    """
    Quantiza para múltiplos de step (em pixels).
    step<=1 => sem quantização.
    """
    step = int(step)
    if step <= 1:
        return int(round(v))
    return int(round(v / step) * step)


def _make_even(n: int, min_n: int = 2) -> int:
    """
    Força paridade (reduz jitter por arredondamento e filtros).
    """
    n = int(n)
    if n < min_n:
        n = min_n
    return n if (n % 2 == 0) else (n - 1)


def _resize_frame_lanczos(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """
    Resize determinístico por PIL/LANCZOS (tende a ser mais estável visualmente).
    """
    if frame.shape[1] == out_w and frame.shape[0] == out_h:
        return frame

    if Image is not None:
        pil = Image.fromarray(frame)
        pil = pil.resize((int(out_w), int(out_h)), resample=Image.ANTIALIAS)
        return np.array(pil)

    # fallback simples sem PIL (não quebra; qualidade inferior)
    y_idx = (np.linspace(0, frame.shape[0] - 1, out_h)).astype(int)
    x_idx = (np.linspace(0, frame.shape[1] - 1, out_w)).astype(int)
    return frame[y_idx][:, x_idx]



# --------------------------
# Dynamic crop + resize (frame-by-frame)
# --------------------------
def _dynamic_crop_resize(
    clip,
    rect_fn: Callable[[float, int, int], Tuple[int, int, int, int]],
    out_w: int,
    out_h: int,
):
    """
    rect_fn(t, frame_w, frame_h) -> (x1,y1,x2,y2) em pixels do frame atual.
    Saída: frame recortado e redimensionado para (out_w,out_h) por frame.
    """
    out_w = int(out_w)
    out_h = int(out_h)

    def _proc(get_frame, t):
        frame = get_frame(t)
        fh = int(frame.shape[0])
        fw = int(frame.shape[1])

        x1, y1, x2, y2 = rect_fn(float(t), fw, fh)

        # clamps finais
        x1 = int(_clamp(x1, 0, fw - 1))
        y1 = int(_clamp(y1, 0, fh - 1))
        x2 = int(_clamp(x2, x1 + 1, fw))
        y2 = int(_clamp(y2, y1 + 1, fh))

        cropped = frame[y1:y2, x1:x2]
        return _resize_frame_lanczos(cropped, out_w, out_h)

    if hasattr(clip, "fl"):
        newclip = clip.fl(_proc, apply_to=["mask"])
        newclip.size = (out_w, out_h)
        return newclip

    if hasattr(clip, "transform"):
        try:
            newclip = clip.transform(lambda gf, t: _proc(gf, t), apply_to=["mask"])
            newclip.size = (out_w, out_h)
            return newclip
        except Exception:
            pass

    raise RuntimeError("MoviePy: não encontrei fl/transform para crop+resize dinâmico.")


# --------------------------
# Zoom-only para IMAGENS (sem pan) — MODO ESTÁVEL
#   Em vez de resize(lambda t:) (que muda dimensões inteiras por frame),
#   fazemos zoom via janela de crop e depois resize para tamanho fixo.
# --------------------------
def apply_zoom_only_image(
    base: ImageClip,
    *,
    out_w: int,
    out_h: int,
    seed_key: str,
    zoom_min: float = 1.01,
    zoom_max: float = 1.06,
    mode: str = "auto",
    quantize_px: int = 1,
    zoom_supersample: int = 1,
    zoom_preblur: float = 0.0,
    force_even: bool = False,
) -> ImageClip:
    """
    Zoom suave sem pan: escala no tempo + crop central para manter 16:9.
    MODO ESTÁVEL: não usa resize(lambda t) para evitar tremido/shimmer.
    Renderiza em resolução maior (supersample), aplica blur opcional, faz downscale com antialias.
    """
    import numpy as np

    if base.duration is None or base.duration <= 0:
        return base

    zoom_min = max(1.0, float(zoom_min))
    zoom_max = max(zoom_min, float(zoom_max))
    ss = max(1, int(zoom_supersample))

    ss_target_w = int(out_w) * ss
    ss_target_h = int(out_h) * ss

    rng = _rng_from_key(seed_key)
    direction = rng.choice(["in", "out"]) if mode == "auto" else mode

    z_hi = rng.uniform(zoom_min, zoom_max)
    z0, z1 = (1.0, z_hi) if direction == "in" else (z_hi, 1.0)

    dur = float(base.duration)

    # upsize imagem base para supersample (mantém cobertura)
    base = base.resize(height=ss_target_h)
    if base.w < ss_target_w:
        base = base.resize(width=ss_target_w)

    # blur inicial opcional (só tenta se cv2 existir)
    if zoom_preblur and float(zoom_preblur) > 0:
        try:
            import cv2  # type: ignore
        except Exception:
            cv2 = None  # type: ignore

        if cv2 is not None:
            sig = float(zoom_preblur)

            def _blur(frame: np.ndarray) -> np.ndarray:
                return cv2.GaussianBlur(frame, (0, 0), sigmaX=sig)

            base = base.fl_image(_blur)

    def scale_at(t: float) -> float:
        a = _smoothstep01(t / dur)
        return z0 + (z1 - z0) * a

    q = max(1, int(quantize_px))

    # Zoom ESTÁVEL: varia o tamanho da janela de crop (em vez de resize no clip)
    def rect_fn(t: float, fw: int, fh: int) -> Tuple[int, int, int, int]:
        z = float(scale_at(float(t)))

        # janela diminui quando z aumenta (zoom in)
        win_w = int(round(ss_target_w / z))
        win_h = int(round(ss_target_h / z))

        if force_even:
            win_w = _make_even(win_w, 2)
            win_h = _make_even(win_h, 2)

        # centraliza
        x1 = (fw - win_w) // 2
        y1 = (fh - win_h) // 2

        x1 = _quantize_px(x1, q)
        y1 = _quantize_px(y1, q)

        if force_even:
            x1 = x1 - (x1 % 2)
            y1 = y1 - (y1 % 2)

        # clamps
        x1 = int(_clamp(x1, 0, max(0, fw - win_w)))
        y1 = int(_clamp(y1, 0, max(0, fh - win_h)))

        return (x1, y1, x1 + win_w, y1 + win_h)

    out = _dynamic_crop_resize(base, rect_fn, out_w=ss_target_w, out_h=ss_target_h)
    out = out.set_duration(dur)

    # downscale final para out_w/out_h com antialias (se ss > 1)
    out = out.resize(width=int(out_w))
    return out


def fit_cover_16x9(clip, target_w: int, target_h: int):
    """
    Resize para cobrir (cover) o target, depois crop central.
    """
    clip = _resize_compat(clip, height=target_h)
    if clip.w < target_w:
        clip = _resize_compat(clip, width=target_w)

    x1 = int((clip.w - target_w) / 2)
    y1 = int((clip.h - target_h) / 2)
    clip = crop_compat(clip, x1=x1, y1=y1, x2=x1 + target_w, y2=y1 + target_h)
    return clip


# --------------------------
# FFmpeg detection + codec
# --------------------------
def ffmpeg_encoders_text() -> str:
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
        return (r.stdout or "") + "\n" + (r.stderr or "")
    except Exception:
        return ""


def ffmpeg_has_encoder(encoder_name: str, encoders_dump: Optional[str] = None) -> bool:
    dump = encoders_dump if encoders_dump is not None else ffmpeg_encoders_text()
    if not dump:
        return False
    return re.search(rf"\b{re.escape(encoder_name)}\b", dump) is not None


def pick_default_vcodec_auto(encoders_dump: Optional[str] = None) -> str:
    sysname = platform.system().lower()

    if "windows" in sysname:
        if ffmpeg_has_encoder("h264_nvenc", encoders_dump):
            return "h264_nvenc"

    if "darwin" in sysname or "mac" in sysname:
        if ffmpeg_has_encoder("h264_videotoolbox", encoders_dump):
            return "h264_videotoolbox"

    return "libx264"


def build_write_kwargs(
    vcodec: str,
    preset: str,
    nvenc_preset: str,
    crf: int,
    bitrate: Optional[str],
    audio_bitrate: str,
    threads: int,
    *,
    test_render: bool = False,
    test_crf: int = 35,
    test_preset: str = "ultrafast",
    test_scale: Optional[int] = 720,
    test_audio_bitrate: str = "96k",
) -> Dict:
    kwargs: Dict = {
        "codec": vcodec,
        "audio_codec": "aac",
        "audio_bitrate": audio_bitrate,
        "threads": threads,
    }

    ffmpeg_params: List[str] = []
    if bitrate:
        kwargs["bitrate"] = bitrate

    if test_render:
        kwargs["codec"] = "libx264"
        kwargs["preset"] = test_preset
        kwargs["audio_bitrate"] = test_audio_bitrate
        ffmpeg_params += ["-crf", str(int(test_crf)), "-pix_fmt", "yuv420p", "-profile:v", "baseline"]
        if test_scale and int(test_scale) > 0:
            ffmpeg_params += ["-vf", f"scale=-2:{int(test_scale)}"]
        kwargs["ffmpeg_params"] = ffmpeg_params
        return kwargs

    if vcodec == "libx264":
        kwargs["preset"] = preset
        ffmpeg_params += ["-crf", str(crf), "-pix_fmt", "yuv420p", "-profile:v", "high"]
    elif vcodec in ("h264_nvenc", "hevc_nvenc"):
        nvenc_ok = {
            "default", "slow", "medium", "fast", "hp", "hq", "bd",
            "ll", "llhq", "llhp", "lossless", "losslesshp"
        }
        preset_final = nvenc_preset if nvenc_preset in nvenc_ok else "hq"
        ffmpeg_params += ["-preset", preset_final, "-rc", "vbr", "-cq", "19", "-pix_fmt", "yuv420p", "-profile:v", "high"]
    elif vcodec in ("h264_videotoolbox", "hevc_videotoolbox"):
        ffmpeg_params += ["-pix_fmt", "yuv420p"]

    if ffmpeg_params:
        kwargs["ffmpeg_params"] = ffmpeg_params
    return kwargs


def ffprobe_stream_info(path: Path) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,sample_aspect_ratio",
            "-of", "default=nk=1:nw=1", str(path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        out = (r.stdout or "").strip().splitlines()
        if len(out) < 3:
            return (None, None, None)
        w = int(out[0]) if out[0].isdigit() else None
        h = int(out[1]) if out[1].isdigit() else None
        sar = out[2].strip() if out[2].strip() else None
        return (w, h, sar)
    except Exception:
        return (None, None, None)


def ffmpeg_cropdetect(path: Path, seconds: float = 1.0) -> Optional[Tuple[int, int, int, int]]:
    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-ss", "0", "-t", str(seconds), "-i", str(path),
            "-vf", "cropdetect=24:16:0", "-f", "null", "-",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        out = (r.stderr or "") + "\n" + (r.stdout or "")
        matches = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", out)
        if not matches:
            return None
        w, h, x, y = map(int, matches[-1])
        if w <= 0 or h <= 0:
            return None
        return (w, h, x, y)
    except Exception:
        return None


# --------------------------
# Cache normalize (opcional)
# --------------------------
def make_cache_name(src: Path) -> str:
    st = src.stat()
    key = f"{src.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", src.stem)
    h = f"{abs(hash(key)) & 0xFFFFFFFF:08x}"
    return f"{safe}__{h}.mp4"


def pick_cache_vcodec(preferred: str) -> str:
    enc_dump = ffmpeg_encoders_text()
    if preferred != "auto" and preferred != "libx264" and ffmpeg_has_encoder(preferred, enc_dump):
        return preferred
    if ffmpeg_has_encoder("h264_nvenc", enc_dump):
        return "h264_nvenc"
    if ffmpeg_has_encoder("h264_videotoolbox", enc_dump):
        return "h264_videotoolbox"
    return "libx264"


def normalize_clip_to_cache(
    src: Path,
    cache_dir: Path,
    force: bool,
    cache_fps: int,
    preferred_vcodec: str,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / make_cache_name(src)

    if not force:
        w, h, _sar = ffprobe_stream_info(src)
        if w == 1920 and h == 1080:
            return src
        if out.exists():
            return out

    vf = "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1"
    vcodec = pick_cache_vcodec(preferred_vcodec)

    if vcodec == "libx264":
        v_params = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    elif vcodec == "h264_nvenc":
        v_params = ["-c:v", "h264_nvenc", "-preset", "fast", "-rc", "vbr", "-cq", "19"]
    elif vcodec == "h264_videotoolbox":
        v_params = ["-c:v", "h264_videotoolbox"]
    else:
        v_params = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-i", str(src),
        "-vf", vf, "-r", str(cache_fps), "-vsync", "cfr",
        *v_params,
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        err = (r.stderr or "").strip()
        raise RuntimeError(f"Falha normalizando clipe:\nSRC: {src}\nOUT: {out}\nFFMPEG:\n{err[-1500:]}")
    return out


# --------------------------
# Render
# --------------------------
def render_video(
    scenes: List[Scene],
    audio_path: Path,
    out_path: Path,
    fps: int,
    vcodec: str,
    preset: str,
    nvenc_preset: str,
    crf: int,
    bitrate: Optional[str],
    audio_bitrate: str,
    debug_ffmpeg: bool,
    cache_clips: bool = False,
    cache_dir: Optional[Path] = None,
    cache_force: bool = False,
    cache_fps: int = 30,
    *,
    test_render: bool = False,
    test_scale: int = 720,
    test_crf: int = 35,
    test_preset: str = "ultrafast",
    test_audio_bitrate: str = "96k",

    # --- Zoom (IMAGENS) ---
    zoom_enabled: bool = True,
    zoom_prob: float = 1.0,
    zoom_min: float = 1.01,
    zoom_max: float = 1.05,
    zoom_mode: str = "auto",          # "auto" | "in" | "out"
    zoom_multiplier: float = 1.0,

    # redução de shimmer/jitter
    zoom_supersample: int = 2,        # 1=off; 2=recomendado; 3=pesado
    zoom_quantize_px: int = 2,        
    zoom_force_even: bool = True,     # recomendado
    zoom_preblur: float = 0.0,        # 0=off; 0.3–0.8 pode ajudar em texturas finas

    # back-compat (se seu make_episode ainda passa isso)
    ken_multiplier: float = 0.0,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    enc_dump = ffmpeg_encoders_text()
    if debug_ffmpeg:
        print("\n[DEBUG] ffmpeg encoders:")
        for name in ("h264_nvenc", "hevc_nvenc", "h264_videotoolbox", "hevc_videotoolbox", "libx264"):
            print(f"  {name}: {ffmpeg_has_encoder(name, enc_dump)}")

    if vcodec == "auto":
        vcodec = pick_default_vcodec_auto(enc_dump)
        print(f"[INFO] vcodec auto selecionado: {vcodec}")

    if (not test_render) and vcodec != "libx264" and not ffmpeg_has_encoder(vcodec, enc_dump):
        print(f"[WARN] ffmpeg não suporta '{vcodec}'. Caindo para libx264.")
        vcodec = "libx264"

    threads = os.cpu_count() or 4
    write_kwargs = build_write_kwargs(
        vcodec=vcodec,
        preset=preset,
        nvenc_preset=nvenc_preset,
        crf=crf,
        bitrate=bitrate,
        audio_bitrate=audio_bitrate,
        threads=threads,
        test_render=test_render,
        test_crf=test_crf,
        test_preset=test_preset,
        test_scale=test_scale,
        test_audio_bitrate=test_audio_bitrate,
    )

    TARGET_W, TARGET_H = 1920, 1080

    SS = int(max(1, zoom_supersample))
    SS_W, SS_H = TARGET_W * SS, TARGET_H * SS

    # multiplica só a parte acima de 1.0
    zoom_min_eff = _apply_multiplier_to_zoom(zoom_min, zoom_multiplier)
    zoom_max_eff = _apply_multiplier_to_zoom(zoom_max, zoom_multiplier)

    audio = None
    timeline: List[VideoFileClip] = []
    bases_to_close: List = []
    final = None

    try:
        audio = AudioFileClip(str(audio_path))

        for i, sc in enumerate(scenes):
            clip_path_use = Path(sc.clip_path)
            ext = clip_path_use.suffix.lower()
            is_img = ext in IMAGE_EXTS

            lead = float(SCENE_CUT_LEAD)
            extra = lead if i < (len(scenes) - 1) else 0.0
            seg_dur = float(sc.duration) + float(extra)

            # Cache só para vídeos
            if cache_clips:
                if cache_dir is None:
                    raise RuntimeError("cache_clips ligado, mas cache_dir é None")
                if not is_img:
                    clip_path_use = normalize_clip_to_cache(
                        src=clip_path_use,
                        cache_dir=cache_dir,
                        force=cache_force,
                        cache_fps=cache_fps,
                        preferred_vcodec=vcodec,
                    )
                    ext = clip_path_use.suffix.lower()

            if is_img:
                base = ImageClip(str(clip_path_use)).set_duration(seg_dur)

                # 1) cover em resolução maior (supersample)
                base = fit_cover_16x9(base, SS_W, SS_H)

                # RNG determinístico por cena
                seed_key = f"{clip_path_use.resolve()}|{i}|{seg_dur:.3f}"
                rng_sel = _rng_from_key(seed_key + "|sel")
                do_zoom = bool(zoom_enabled) and (rng_sel.random() < float(zoom_prob))

                if do_zoom:
                    q_ss = max(1, int(zoom_quantize_px))

                    base = apply_zoom_only_image(
                        base,
                        out_w=SS_W,
                        out_h=SS_H,
                        seed_key=seed_key + "|zoom",
                        zoom_min=zoom_min_eff,
                        zoom_max=zoom_max_eff,
                        mode=zoom_mode,          # "auto" alterna in/out por imagem
                        quantize_px=q_ss,
                        force_even=bool(zoom_force_even),
                    )

                # 2) blur leve antes do downscale (opcional)
                if zoom_preblur and float(zoom_preblur) > 0 and _gaussian_blur_fx is not None:
                    try:
                        base = base.fx(_gaussian_blur_fx, float(zoom_preblur))
                    except Exception:
                        pass

                # 3) downscale final para 1080p (crítico para reduzir shimmer)
                base = _resize_compat(base, height=TARGET_H)
                if base.w != TARGET_W:
                    base = _resize_compat(base, width=TARGET_W)

                seg = base

            else:
                base = VideoFileClip(str(clip_path_use), audio=False)

                if not test_render:
                    cd = ffmpeg_cropdetect(clip_path_use, seconds=1.0)
                    if cd:
                        w, h, x, y = cd
                        base = crop_compat(base, x1=x, y1=y, x2=x + w, y2=y + h)
                    base = fit_cover_16x9(base, TARGET_W, TARGET_H)
                else:
                    base = fit_cover_16x9(base, TARGET_W, TARGET_H)

                seg = loop_or_trim(base, seg_dur)

            timeline.append(seg)
            bases_to_close.append(base)

        if not timeline:
            raise RuntimeError("Timeline vazia: nenhum segmento foi gerado.")

        method = "compose" if (test_render or SCENE_CUT_LEAD > 0) else "chain"
        final = concatenate_videoclips(
            timeline,
            method=method,
            padding=(-float(SCENE_CUT_LEAD) if SCENE_CUT_LEAD > 0 else 0.0),
        )

        try:
            a = audio.subclip(0, final.duration)
        except Exception:
            a = audio

        if hasattr(final, "set_audio"):
            final = final.set_audio(a)
        elif hasattr(final, "with_audio"):
            final = final.with_audio(a)
        else:
            raise RuntimeError("MoviePy: não encontrei set_audio/with_audio para anexar áudio.")

        final.write_videofile(
            str(out_path),
            fps=fps,
            **write_kwargs,
            verbose=debug_ffmpeg,
            logger="bar" if not debug_ffmpeg else None,
        )

        if (not out_path.exists()) or out_path.stat().st_size < 1024:
            raise RuntimeError(f"Falha: arquivo não foi criado corretamente: {out_path}")

    finally:
        if final is not None:
            try:
                final.close()
            except Exception:
                pass

        for c in timeline:
            try:
                c.close()
            except Exception:
                pass

        if audio is not None:
            try:
                audio.close()
            except Exception:
                pass

        for b in bases_to_close:
            try:
                b.close()
            except Exception:
                pass
