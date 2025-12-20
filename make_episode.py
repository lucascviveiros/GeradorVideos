import argparse
import os
import platform
import random
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Import robusto (MoviePy 1.x vs 2.x + diferentes layouts)
try:
    from moviepy.editor import AudioFileClip, VideoFileClip, concatenate_videoclips
except Exception:
    from moviepy import AudioFileClip, VideoFileClip, concatenate_videoclips

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".m4v", ".webm"}


# --------------------------
# Utils
# --------------------------
def _norm(s: str) -> str:
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


def resolve_path(user_path: str, must_exist: bool = False, is_dir: bool = False) -> Path:
    p = Path(user_path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p)
    p = p.resolve(strict=False)

    if must_exist:
        if is_dir and not p.is_dir():
            raise RuntimeError(f"Pasta não encontrada: {p}")
        if not is_dir and not p.is_file():
            raise RuntimeError(f"Arquivo não encontrado: {p}")
    return p


def debug_path_report(label: str, p: Path):
    print(f"\n[DEBUG] {label}")
    print(f"  cwd: {Path.cwd()}")
    print(f"  path: {p}")
    print(f"  exists: {p.exists()} | is_file: {p.is_file()} | is_dir: {p.is_dir()}")
    if p.parent.exists():
        try:
            entries = list(p.parent.iterdir())
            print(f"  parent: {p.parent} (itens: {len(entries)})")
            for e in entries[:15]:
                print(f"    - {e.name}")
        except Exception as ex:
            print(f"  não consegui listar parent: {ex}")


def _subclip_compat(clip: VideoFileClip, start: float, end: float) -> VideoFileClip:
    """
    Compat MoviePy 1.x (subclip) e 2.x (subclipped).
    """
    if hasattr(clip, "subclip"):
        return clip.subclip(start, end)
    if hasattr(clip, "subclipped"):
        return clip.subclipped(start, end)
    raise RuntimeError("MoviePy: não encontrei subclip/subclipped nesta versão.")


# --------------------------
# FFmpeg detection + codec selection
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
    # procura linha contendo o encoder (evita falso positivo em help)
    return re.search(rf"\b{re.escape(encoder_name)}\b", dump) is not None


def pick_default_vcodec_auto(encoders_dump: Optional[str] = None) -> str:
    """
    Heurística:
      - Windows: preferir h264_nvenc se disponível
      - macOS: preferir h264_videotoolbox se disponível
      - fallback: libx264
    """
    sysname = platform.system().lower()

    if "windows" in sysname:
        if ffmpeg_has_encoder("h264_nvenc", encoders_dump):
            return "h264_nvenc"
        # opcional: hevc_nvenc se preferir H.265
        # if ffmpeg_has_encoder("hevc_nvenc", encoders_dump): return "hevc_nvenc"

    if "darwin" in sysname or "mac" in sysname:
        if ffmpeg_has_encoder("h264_videotoolbox", encoders_dump):
            return "h264_videotoolbox"
        # if ffmpeg_has_encoder("hevc_videotoolbox", encoders_dump): return "hevc_videotoolbox"

    return "libx264"


from typing import Dict, List, Optional


def build_write_kwargs(
    vcodec: str,
    preset: str,
    nvenc_preset: str,
    crf: int,
    bitrate: Optional[str],
    audio_bitrate: str,
    threads: int,
) -> Dict:
    """
    Monta kwargs para video.write_videofile.

    Regras importantes:
    - MoviePy só repassa corretamente alguns argumentos nativos.
    - Para NVENC e ajustes avançados, SEMPRE usar ffmpeg_params.
    - Presets precisam ser compatíveis com o encoder, senão o FFmpeg aborta.
    """

    kwargs: Dict = {
        "codec": vcodec,
        "audio_codec": "aac",
        "audio_bitrate": audio_bitrate,
        "threads": threads,
    }

    ffmpeg_params: List[str] = []

    # Bitrate fixo (opcional)
    if bitrate:
        kwargs["bitrate"] = bitrate

    # -----------------------------
    # libx264 (CPU)
    # -----------------------------
    if vcodec == "libx264":
        # Presets válidos x264:
        # ultrafast/superfast/veryfast/faster/fast/medium/slow/slower/veryslow
        kwargs["preset"] = preset

        ffmpeg_params += [
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
        ]

    # -----------------------------
    # NVENC (NVIDIA GPU)
    # -----------------------------
    elif vcodec in ("h264_nvenc", "hevc_nvenc"):
        # Presets válidos para h264_nvenc / hevc_nvenc
        # NÃO usar p1..p7 aqui
        nvenc_presets_validos = {
            "default",
            "slow",
            "medium",
            "fast",
            "hp",
            "hq",
            "bd",
            "ll",
            "llhq",
            "llhp",
            "lossless",
            "losslesshp",
        }

        preset_final = nvenc_preset if nvenc_preset in nvenc_presets_validos else "hq"

        ffmpeg_params += [
            "-preset", preset_final,
            "-rc", "vbr",
            "-cq", "19",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
        ]

    # -----------------------------
    # VideoToolbox (macOS)
    # -----------------------------
    elif vcodec in ("h264_videotoolbox", "hevc_videotoolbox"):
        # Presets são limitados; foco é compatibilidade
        ffmpeg_params += [
            "-pix_fmt", "yuv420p"
        ]

    # -----------------------------
    # Finalização
    # -----------------------------
    if ffmpeg_params:
        kwargs["ffmpeg_params"] = ffmpeg_params

    return kwargs


# --------------------------
# Tags + paragraphs
# --------------------------
def load_tags_yaml(path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    if not path.is_file():
        raise RuntimeError(f"tags.yaml não encontrado: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    tags: Dict[str, List[str]] = data.get("tags", {})
    if not isinstance(tags, dict) or not tags:
        raise RuntimeError("tags.yaml não tem a chave 'tags' ou está vazia.")

    tag_order = list(tags.keys())
    tags_norm = {k: [_norm(str(x)) for x in v] for k, v in tags.items()}
    return tag_order, tags_norm


def read_paragraphs(text_path: Path, debug_paths: bool = False) -> List[str]:
    if debug_paths:
        debug_path_report("text_path", text_path)

    if not text_path.is_file():
        raise RuntimeError(f"Roteiro não encontrado: {text_path}")

    raw = text_path.read_text(encoding="utf-8").strip()
    parts = re.split(r"\n\s*\n+", raw)

    paras: List[str] = []
    for part in parts:
        part = part.strip()
        if part:
            paras.append(re.sub(r"\s+", " ", part))

    if not paras:
        raise RuntimeError("Nenhum parágrafo encontrado no roteiro.")
    return paras


def word_count(s: str) -> int:
    return len(re.findall(r"\b[\wÀ-ÿ]+\b", s))


def score_tag(paragraph: str, keywords: List[str]) -> int:
    p = _norm(paragraph)
    score = 0
    for kw in keywords:
        if kw and kw in p:
            score += 1
    return score


def choose_tag(paragraph: str, tag_order: List[str], tags: Dict[str, List[str]], fallback: str) -> str:
    best_tag = fallback
    best_score = -1
    for tag in tag_order:
        s = score_tag(paragraph, tags.get(tag, []))
        if s > best_score:
            best_score = s
            best_tag = tag
    return best_tag if best_score > 0 else fallback


def index_clips(clips_root: Path) -> Dict[str, List[Path]]:
    if not clips_root.is_dir():
        raise RuntimeError(f"Pasta raiz de clipes não existe: {clips_root}")

    index: Dict[str, List[Path]] = {}
    for sub in clips_root.iterdir():
        if not sub.is_dir():
            continue
        files = [f for f in sub.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS]
        files.sort(key=lambda x: x.name.lower())
        if files:
            index[sub.name] = files

    if not index:
        raise RuntimeError(
            f"Nenhum vídeo encontrado em subpastas de {clips_root}.\n"
            f"Esperado: {clips_root}/tag1/*.mp4, {clips_root}/tag2/*.mp4 ..."
        )
    return index


def pick_clip(clips: List[Path], rng: random.Random, last_used: Optional[Path]) -> Path:
    if not clips:
        raise RuntimeError("Lista de clipes vazia.")

    # tenta não repetir imediatamente
    if last_used and len(clips) > 1:
        for _ in range(8):
            c = rng.choice(clips)
            if c != last_used:
                return c
    return rng.choice(clips)


def loop_or_trim(clip: VideoFileClip, target_dur: float) -> VideoFileClip:
    if target_dur <= 0:
        return _subclip_compat(clip, 0, 0.01)

    if clip.duration >= target_dur:
        return _subclip_compat(clip, 0, target_dur)

    reps = int(target_dur // clip.duration) + 1
    looped = concatenate_videoclips([clip] * reps)
    return _subclip_compat(looped, 0, target_dur)


@dataclass
class Scene:
    tag: str
    text: str
    duration: float
    clip_path: Path


def build_scenes(
    paras: List[str],
    audio_dur: float,
    tag_order: List[str],
    tags: Dict[str, List[str]],
    clips_root: Path,
    fallback_tag: str,
    min_scene: float,
    max_scene: float,
    seed: int,
    single_folder: bool = False,
    single_folder_value: Optional[str] = None,
) -> List[Scene]:
    rng = random.Random(seed)

    counts = [max(1, word_count(p)) for p in paras]
    total_words = sum(counts)

    raw_durs = [(c / total_words) * audio_dur for c in counts]
    clamped = [min(max(d, min_scene), max_scene) for d in raw_durs]
    sum_clamped = sum(clamped)
    if sum_clamped <= 0:
        raise RuntimeError("Soma de durações inválida (sum_clamped <= 0).")

    scale = audio_dur / sum_clamped
    final_durs = [d * scale for d in clamped]

    clips_index = index_clips(clips_root)

    # Single folder mode
    if single_folder:
        if single_folder_value:
            p = Path(single_folder_value).expanduser()
            folder_path = p if p.is_dir() else (clips_root / single_folder_value)
        else:
            folder_path = clips_root / fallback_tag

        folder_path = folder_path.resolve(strict=False)
        if not folder_path.is_dir():
            raise RuntimeError(f"--single_folder ativado, mas pasta não existe: {folder_path}")

        single_clips = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS]
        single_clips.sort(key=lambda x: x.name.lower())
        if not single_clips:
            raise RuntimeError(f"--single_folder ativado, mas não achei vídeos em: {folder_path}")

        bucket = folder_path.name
        last_used: Optional[Path] = None
        scenes: List[Scene] = []

        for ptxt, dur in zip(paras, final_durs):
            clip_path = pick_clip(single_clips, rng, last_used)
            last_used = clip_path
            scenes.append(Scene(tag=bucket, text=ptxt, duration=float(dur), clip_path=clip_path))

        drift = audio_dur - sum(s.duration for s in scenes)
        if scenes and abs(drift) > 0.02:
            scenes[-1].duration = max(0.3, scenes[-1].duration + drift)

        return scenes

    # Normal mode (tags)
    scenes: List[Scene] = []
    last_used_per_tag: Dict[str, Optional[Path]] = {}

    for ptxt, dur in zip(paras, final_durs):
        tag = choose_tag(ptxt, tag_order, tags, fallback=fallback_tag)
        clips = clips_index.get(tag) or clips_index.get(fallback_tag)
        if not clips:
            raise RuntimeError(
                f"Não achei clipes para tag '{tag}' nem para fallback '{fallback_tag}'. "
                f"Tags disponíveis: {sorted(clips_index.keys())}"
            )
        if tag not in clips_index:
            tag = fallback_tag

        clip_path = pick_clip(clips, rng, last_used_per_tag.get(tag))
        last_used_per_tag[tag] = clip_path

        scenes.append(Scene(tag=tag, text=ptxt, duration=float(dur), clip_path=clip_path))

    drift = audio_dur - sum(s.duration for s in scenes)
    if scenes and abs(drift) > 0.02:
        scenes[-1].duration = max(0.3, scenes[-1].duration + drift)

    return scenes


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
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    enc_dump = ffmpeg_encoders_text()
    if debug_ffmpeg:
        print("\n[DEBUG] ffmpeg encoders: procurando suporte...")
        for name in ("h264_nvenc", "hevc_nvenc", "h264_videotoolbox", "hevc_videotoolbox", "libx264"):
            print(f"  {name}: {ffmpeg_has_encoder(name, enc_dump)}")

    # auto codec selection
    if vcodec == "auto":
        vcodec = pick_default_vcodec_auto(enc_dump)
        print(f"[INFO] vcodec auto selecionado: {vcodec}")

    # valida codec escolhido (se não suportado, cai para libx264)
    if vcodec != "libx264" and not ffmpeg_has_encoder(vcodec, enc_dump):
        print(f"[WARN] ffmpeg não suporta '{vcodec}'. Caindo para libx264.")
        vcodec = "libx264"

    threads = os.cpu_count() or 4
    # NVENC / VideoToolbox não se beneficia de threads do mesmo jeito — mas não atrapalha.
    write_kwargs = build_write_kwargs(
        vcodec=vcodec,
        preset=preset,
        nvenc_preset=nvenc_preset,
        crf=crf,
        bitrate=bitrate,
        audio_bitrate=audio_bitrate,
        threads=threads,
    )

    audio = AudioFileClip(str(audio_path))
    timeline: List[VideoFileClip] = []

    try:
        for sc in scenes:
            base = VideoFileClip(str(sc.clip_path), audio=False)
            #base = VideoFileClip(str(sc.clip_path))
            seg = loop_or_trim(base, sc.duration)
            timeline.append(seg)

        video = concatenate_videoclips(timeline, method="compose").set_audio(audio)

        video.write_videofile(
            str(out_path),
            fps=fps,
            **write_kwargs,
        )

        try:
            video.close()
        except Exception:
            pass

    finally:
        try:
            audio.close()
        except Exception:
            pass
        for c in timeline:
            try:
                c.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(
        description="Auto-montagem de vídeo com roteiro + áudio + clipes em pastas por tag (ou pasta única)."
    )
    ap.add_argument("--audio", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--tags", default="tags.yaml")
    ap.add_argument("--out", default="out/episode.mp4")
    ap.add_argument("--fallback", default="abstract_dark")
    ap.add_argument("--min_scene", type=float, default=4.5)
    ap.add_argument("--max_scene", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print_plan", action="store_true")
    ap.add_argument("--debug_paths", action="store_true")

    ap.add_argument("--single_folder", action="store_true")
    ap.add_argument("--single_folder_value", default=None)

    # ✅ Render acceleration options
    ap.add_argument(
        "--vcodec",
        default="auto",
        help="auto|libx264|h264_nvenc|hevc_nvenc|h264_videotoolbox|hevc_videotoolbox",
    )
    ap.add_argument("--preset", default="veryfast", help="Preset do libx264 (ultrafast..veryslow).")
    ap.add_argument("--nvenc_preset", default="p4", help="Preset NVENC (p1..p7). Recomendo p4.")
    ap.add_argument("--crf", type=int, default=20, help="CRF para libx264 (18=melhor, 23=menor).")
    ap.add_argument("--bitrate", default=None, help="Bitrate de vídeo (ex: 6M). Se vazio, automático.")
    ap.add_argument("--audio_bitrate", default="192k", help="Bitrate de áudio AAC (ex: 128k, 192k).")
    ap.add_argument("--debug_ffmpeg", action="store_true", help="Mostra se ffmpeg suporta NVENC/VideoToolbox etc.")

    args = ap.parse_args()

    audio_p = resolve_path(args.audio, must_exist=True, is_dir=False)
    text_p = resolve_path(args.text, must_exist=True, is_dir=False)
    clips_p = resolve_path(args.clips, must_exist=True, is_dir=True)
    tags_p = resolve_path(args.tags, must_exist=True, is_dir=False)
    out_p = resolve_path(args.out, must_exist=False, is_dir=False)

    if args.debug_paths:
        debug_path_report("audio", audio_p)
        debug_path_report("text", text_p)
        debug_path_report("clips", clips_p)
        debug_path_report("tags", tags_p)
        debug_path_report("out", out_p)

    tag_order, tags = load_tags_yaml(tags_p)
    paras = read_paragraphs(text_p, debug_paths=args.debug_paths)

    a = AudioFileClip(str(audio_p))
    audio_dur = a.duration
    a.close()

    scenes = build_scenes(
        paras=paras,
        audio_dur=audio_dur,
        tag_order=tag_order,
        tags=tags,
        clips_root=clips_p,
        fallback_tag=args.fallback,
        min_scene=args.min_scene,
        max_scene=args.max_scene,
        seed=args.seed,
        single_folder=args.single_folder,
        single_folder_value=args.single_folder_value,
    )

    if args.print_plan:
        total = 0.0
        for i, sc in enumerate(scenes, 1):
            total += sc.duration
            print(f"{i:02d} | {sc.duration:5.2f}s | {sc.tag:<16} | {sc.clip_path.name}")
        print(f"Total cenas: {len(scenes)} | Duração áudio: {audio_dur:.2f}s | Duração vídeo: {total:.2f}s")
        return

    render_video(
        scenes=scenes,
        audio_path=audio_p,
        out_path=out_p,
        fps=args.fps,
        vcodec=args.vcodec,
        preset=args.preset,
        nvenc_preset=args.nvenc_preset,
        crf=args.crf,
        bitrate=args.bitrate,
        audio_bitrate=args.audio_bitrate,
        debug_ffmpeg=args.debug_ffmpeg,
    )

    print(f"OK: {out_p}")


if __name__ == "__main__":
    main()



#py -3.10 make_episode.py --audio "audio\ep001.mp3" --text "narrativa\ep001.txt" --clips "clips" --single_folder --single_folder_value abstract_dark --vcodec auto --debug_ffmpeg


#py -3.10 make_episode.py --audio "audio\ep001.mp3" --text "narrativa\ep001.txt" --clips "clips" --single_folder --single_folder_value abstract_dark --vcodec h264_videotoolbox --debug_ffmpeg


#py -3.10 make_episode.py --audio "audio\ep001.mp3" --text "narrativa\ep001.txt" --clips "clips" --single_folder --single_folder_value abstract_dark --vcodec h264_nvenc --nvenc_preset fast --debug_ffmpeg