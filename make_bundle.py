#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def run(cmd: List[str]) -> None:
    print("\n[RUN]", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"Falha ao executar (code={r.returncode}): {' '.join(cmd)}")


def parse_episodes(s: str) -> List[Tuple[str, int]]:
    """
    Aceita:
      - "ep001,ep002,ep003"
      - "1-3"
      - "001-003"
      - "ep001-ep010"
    Retorna lista de tuplas: (ep_id_str, ep_num_int)
    """
    s = s.strip()
    if not s:
        raise RuntimeError("--episodes vazio")

    # lista separada por vírgula
    if "," in s:
        items = [x.strip() for x in s.split(",") if x.strip()]
        out: List[Tuple[str, int]] = []
        for it in items:
            m = re.match(r"^ep?(\d+)$", it, re.IGNORECASE)
            if not m:
                raise RuntimeError(f"Episódio inválido na lista: '{it}' (use ep001 ou 001)")
            n = int(m.group(1))
            out.append((f"ep{n:03d}", n))
        return out

    # range
    m = re.match(r"^(ep?\d+)\s*-\s*(ep?\d+)$", s, re.IGNORECASE)
    if m:
        a = re.match(r"^ep?(\d+)$", m.group(1), re.IGNORECASE)
        b = re.match(r"^ep?(\d+)$", m.group(2), re.IGNORECASE)
        if not a or not b:
            raise RuntimeError(f"Range inválido: '{s}'")
        start = int(a.group(1))
        end = int(b.group(1))
        if end < start:
            raise RuntimeError(f"Range inválido (fim < início): '{s}'")
        return [(f"ep{i:03d}", i) for i in range(start, end + 1)]

    # range numérico simples "1-10"
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", s)
    if m:
        start = int(m.group(1))
        end = int(m.group(2))
        if end < start:
            raise RuntimeError(f"Range inválido (fim < início): '{s}'")
        return [(f"ep{i:03d}", i) for i in range(start, end + 1)]

    # único
    m = re.match(r"^ep?(\d+)$", s, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        return [(f"ep{n:03d}", n)]

    raise RuntimeError(f"--episodes inválido: '{s}' (ex: ep001,ep002 ou 1-3 ou ep001-ep010)")


def must_file(p: Path, label: str) -> None:
    if not p.is_file():
        raise RuntimeError(f"{label} não encontrado: {p}")


def must_dir(p: Path, label: str) -> None:
    if not p.is_dir():
        raise RuntimeError(f"{label} não encontrado: {p}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bundle: gera áudio (XTTS) e vídeo (MoviePy) para múltiplos episódios e idiomas."
    )

    ap.add_argument("--episodes", required=True, help="ex: ep001,ep002,ep003 ou 1-3 ou ep001-ep010")
    ap.add_argument("--langs", default="pt,en,es", help="ex: pt,en,es ou pt")

    ap.add_argument("--narr_dir", default="narrativa", help="Pasta de textos (estrutura + epXXX_lang.txt)")
    ap.add_argument("--voices_dir", default="voices", help="Pasta de vozes (voice_pt.mp3 etc.)")
    ap.add_argument("--audio_dir", default="audio", help="Saída de áudios")
    ap.add_argument("--out_dir", default="out", help="Saída de vídeos")
    ap.add_argument("--clips", required=True, help="Pasta raiz de clipes por tag (subpastas)")

    # NOVO: padrão do arquivo de estrutura com TAGS B-ROLL
    # {id}  -> ep001
    # {num} -> 1
    # {num:03d} -> 001
    ap.add_argument(
        "--structure_pattern",
        default="estrutura_{num:03d}.txt",
        help="Nome do arquivo de estrutura/tagueado. Ex: estrutura_{num:03d}.txt",
    )

    # Áudio (repasse para make_audio.py)
    ap.add_argument("--gpu", action="store_true", help="Ativa GPU no TTS (make_audio.py)")

    # Vídeo (repasse para make_episode.py)
    ap.add_argument("--vcodec", default="auto")
    ap.add_argument("--nvenc_preset", default="hq")
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--audio_bitrate", default="192k")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_scene", type=float, default=4.5)
    ap.add_argument("--max_scene", type=float, default=10.0)

    # extras
    ap.add_argument("--debug_ffmpeg", action="store_true")
    ap.add_argument("--print_plan", action="store_true")
    ap.add_argument("--cache_clips", action="store_true")
    ap.add_argument("--cache_dir", default="clips_cache")
    ap.add_argument("--cache_force", action="store_true")
    ap.add_argument("--cache_fps", type=int, default=30)

    args = ap.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    if not langs:
        raise RuntimeError("--langs inválido")

    narr_dir = Path(args.narr_dir)
    voices_dir = Path(args.voices_dir)
    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out_dir)
    clips_dir = Path(args.clips)

    must_dir(narr_dir, "narr_dir")
    must_dir(voices_dir, "voices_dir")
    must_dir(clips_dir, "clips")

    audio_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # valida vozes fixas
    for lang in langs:
        voice = voices_dir / f"voice_{lang}.wav"
        must_file(voice, f"Voz ({lang})")

    episodes = parse_episodes(args.episodes)

    for ep_id, ep_num in episodes:
        # >>> AQUI mudou: usamos a estrutura com TAGS B-ROLL, não o epXXX_pt.txt <<<
        structure_name = args.structure_pattern.format(id=ep_id, num=ep_num)
        structure_path = narr_dir / structure_name
        must_file(structure_path, f"Estrutura base TAGs ({ep_id})")

        # 1) AUDIO por idioma (pula se já existir)
        for lang in langs:
            txt = narr_dir / f"{ep_id}_{lang}.txt"
            must_file(txt, f"Texto falado ({ep_id}/{lang})")

            audio_out = audio_dir / f"{ep_id}_{lang}.mp3"

            if audio_out.is_file() and audio_out.stat().st_size > 1024:
                print(f"[SKIP] Áudio já existe: {audio_out}")
                continue

            cmd_audio = [
                sys.executable, "make_audio.py",
                "--episode", ep_id,
                "--narr_dir", str(narr_dir),
                "--out_dir", str(audio_dir),
                "--langs", lang,
                "--voices_dir", str(voices_dir),
            ]
            if args.gpu:
                cmd_audio.append("--gpu")

            # Repasse do preset FAST
            cmd_audio += [
                "--segments",
                "--normalize_wavs",
                "--max_chars", "420",
                "--min_chars", "140",
                "--mp3_vbr_q", "2",
                "--speed", "1.20",
            ]

            run(cmd_audio)

        # 2) VIDEO por idioma
        for lang in langs:
            audio_path = audio_dir / f"{ep_id}_{lang}.mp3"
            must_file(audio_path, f"Áudio gerado ({ep_id}/{lang})")

            out_path = out_dir / f"{ep_id}_{lang}.mp4"

            cmd_video = [
                sys.executable, "make_episode.py",
                "--audio", str(audio_path),
                "--langs", lang,
                "--text", str(structure_path),
                "--clips", str(clips_dir),
                "--out", str(out_path),

                "--vcodec", args.vcodec,
                "--preset", args.preset,
                "--nvenc_preset", args.nvenc_preset,
                "--crf", str(args.crf),

                "--audio_bitrate", args.audio_bitrate,
                "--fps", str(args.fps),
                "--seed", str(args.seed),
                "--min_scene", str(args.min_scene),
                "--max_scene", str(args.max_scene),
            ]

            if args.debug_ffmpeg:
                cmd_video.append("--debug_ffmpeg")

            if args.print_plan:
                cmd_video.append("--print_plan")

            if args.cache_clips:
                cmd_video += [
                    "--cache_clips",
                    "--cache_dir", args.cache_dir,
                    "--cache_fps", str(args.cache_fps),
                ]
                if args.cache_force:
                    cmd_video.append("--cache_force")

            run(cmd_video)

        print(f"\n[OK] Episódio finalizado: {ep_id} (langs: {','.join(langs)})")

    print("\n✅ Bundle finalizado: todos os episódios processados.")


if __name__ == "__main__":
    main()


#py -3.10 .\make_bundle.py --episodes ep001 --narr_dir narrativa --audio_dir audio --out_dir out --langs pt --voices_dir voices --gpu --clips clips --vcodec h264_nvenc --nvenc_preset fast
