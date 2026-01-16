#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- Pillow/MoviePy compatibility patch (Pillow 10+ removed Image.ANTIALIAS) ---
try:
    from PIL import Image
    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
except Exception:
    pass

try:
    from moviepy import AudioFileClip, VideoFileClip, concatenate_videoclips
except Exception:
    from moviepy.editor import AudioFileClip, VideoFileClip, concatenate_videoclips

from episode_plan import Scene

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".m4v", ".webm"}


# --------------------------
# MoviePy compat
# --------------------------
def _subclip_compat(clip: VideoFileClip, start: float, end: float) -> VideoFileClip:
    if hasattr(clip, "subclip"):
        return clip.subclip(start, end)
    if hasattr(clip, "subclipped"):
        return clip.subclipped(start, end)
    raise RuntimeError("MoviePy: não encontrei subclip/subclipped nesta versão.")


def _resize_compat(clip: VideoFileClip, *, width: Optional[int] = None, height: Optional[int] = None) -> VideoFileClip:
    if hasattr(clip, "resize"):
        return clip.resize(width=width, height=height)
    if hasattr(clip, "resized"):
        return clip.resized(width=width, height=height)
    raise RuntimeError("MoviePy: não encontrei resize/resized nesta versão.")


def crop_compat(clip: VideoFileClip, x1: int, y1: int, x2: int, y2: int) -> VideoFileClip:
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

    if vcodec == "libx264":
        kwargs["preset"] = preset
        ffmpeg_params += ["-crf", str(crf), "-pix_fmt", "yuv420p", "-profile:v", "high"]
    elif vcodec in ("h264_nvenc", "hevc_nvenc"):
        nvenc_ok = {
            "default","slow","medium","fast","hp","hq","bd","ll","llhq","llhp","lossless","losslesshp"
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
            "ffprobe","-v","error","-select_streams","v:0",
            "-show_entries","stream=width,height,sample_aspect_ratio",
            "-of","default=nk=1:nw=1", str(path),
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
            "ffmpeg","-hide_banner","-ss","0","-t",str(seconds),"-i",str(path),
            "-vf","cropdetect=24:16:0","-f","null","-",
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
        v_params = ["-c:v","libx264","-preset","veryfast","-crf","20"]
    elif vcodec == "h264_nvenc":
        v_params = ["-c:v","h264_nvenc","-preset","fast","-rc","vbr","-cq","19"]
    elif vcodec == "h264_videotoolbox":
        v_params = ["-c:v","h264_videotoolbox"]
    else:
        v_params = ["-c:v","libx264","-preset","veryfast","-crf","20"]

    cmd = [
        "ffmpeg","-y","-hide_banner","-i",str(src),
        "-vf",vf, "-r",str(cache_fps), "-vsync","cfr",
        *v_params,
        "-c:a","aac","-b:a","192k",
        "-movflags","+faststart",
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
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    enc_dump = ffmpeg_encoders_text()
    if debug_ffmpeg:
        print("\n[DEBUG] ffmpeg encoders:")
        for name in ("h264_nvenc","hevc_nvenc","h264_videotoolbox","hevc_videotoolbox","libx264"):
            print(f"  {name}: {ffmpeg_has_encoder(name, enc_dump)}")

    if vcodec == "auto":
        vcodec = pick_default_vcodec_auto(enc_dump)
        print(f"[INFO] vcodec auto selecionado: {vcodec}")

    if vcodec != "libx264" and not ffmpeg_has_encoder(vcodec, enc_dump):
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
    )

    TARGET_W, TARGET_H = 1920, 1080

    audio = None
    timeline: List[VideoFileClip] = []
    bases_to_close: List[VideoFileClip] = []
    final = None

    try:
        audio = AudioFileClip(str(audio_path))

        for sc in scenes:
            clip_path_use = sc.clip_path
            if cache_clips:
                if cache_dir is None:
                    raise RuntimeError("cache_clips ligado, mas cache_dir é None")
                clip_path_use = normalize_clip_to_cache(
                    src=sc.clip_path,
                    cache_dir=cache_dir,
                    force=cache_force,
                    cache_fps=cache_fps,
                    preferred_vcodec=vcodec,
                )
                base = VideoFileClip(str(clip_path_use), audio=False)
            else:
                base = VideoFileClip(str(clip_path_use), audio=False)

                cd = ffmpeg_cropdetect(clip_path_use, seconds=1.0)
                if cd:
                    w, h, x, y = cd
                    base = crop_compat(base, x1=x, y1=y, x2=x + w, y2=y + h)

                base = _resize_compat(base, height=TARGET_H)
                if base.w < TARGET_W:
                    base = _resize_compat(base, width=TARGET_W)

                x1 = int((base.w - TARGET_W) / 2)
                y1 = int((base.h - TARGET_H) / 2)
                base = crop_compat(base, x1=x1, y1=y1, x2=x1 + TARGET_W, y2=y1 + TARGET_H)

            seg = loop_or_trim(base, sc.duration)
            timeline.append(seg)
            bases_to_close.append(base)

        if not timeline:
            raise RuntimeError("Timeline vazia: nenhum segmento foi gerado.")

        final = concatenate_videoclips(timeline, method="chain").set_audio(audio)

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
            try: final.close()
            except Exception: pass

        for c in timeline:
            try: c.close()
            except Exception: pass

        if audio is not None:
            try: audio.close()
            except Exception: pass

        for b in bases_to_close:
            try: b.close()
            except Exception: pass
