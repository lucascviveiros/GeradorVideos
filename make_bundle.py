#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bundle runner:
1) Gera áudio por idioma (make_audio.py)
2) Gera vídeo por idioma usando o áudio recém-criado (make_episode.py)

Objetivo:
- 1 comando
- pipeline determinístico
- pronto para escalar com políticas YouTube 2026
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def run(cmd: List[str]):
    print("\n[RUN]", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"Falha ao executar: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser("make_bundle.py")

    ap.add_argument("--episode", required=True, help="Ex: ep001")
    ap.add_argument("--langs", default="pt,en,es")
    ap.add_argument("--narr_dir", default="narrativa")
    ap.add_argument("--voices_dir", default="voices")
    ap.add_argument("--clips", required=True)
    ap.add_argument("--structure", required=True, help="estrutura editorial (ex: estrutura_001.txt)")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--audio_dir", default="audio")

    # audio
    ap.add_argument("--gpu", action="store_true")

    # video
    ap.add_argument("--vcodec", default="auto")
    ap.add_argument("--nvenc_preset", default="hq")
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--crf", default="20")

    args = ap.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    ep = args.episode

    narr_dir = Path(args.narr_dir)
    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out_dir)

    audio_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1) AUDIO
    # -------------------------
    for lang in langs:
        txt = narr_dir / f"{ep}_{lang}.txt"
        voice = Path(args.voices_dir) / f"voice_{lang}.mp3"

        if not txt.is_file():
            raise RuntimeError(f"Texto não encontrado: {txt}")
        if not voice.is_file():
            raise RuntimeError(f"Voz não encontrada: {voice}")

        cmd_audio = [
            sys.executable, "make_audio.py",
            "--episode", ep,
            "--langs", lang,
            "--narr_dir", str(narr_dir),
            "--voices_dir", args.voices_dir,
            "--out_dir", str(audio_dir),
        ]

        if args.gpu:
            cmd_audio.append("--gpu")

        run(cmd_audio)

    # -------------------------
    # 2) VIDEO
    # -------------------------
    for lang in langs:
        audio = audio_dir / f"{ep}_{lang}.mp3"
        out = out_dir / f"{ep}_{lang}.mp4"

        if not audio.is_file():
            raise RuntimeError(f"Áudio não encontrado: {audio}")

        cmd_video = [
            sys.executable, "make_episode.py",
            "--audio", str(audio),
            "--langs", lang,
            "--text", args.structure,
            "--clips", args.clips,
            "--out", str(out),
            "--vcodec", args.vcodec,
            "--preset", args.preset,
            "--nvenc_preset", args.nvenc_preset,
            "--crf", str(args.crf),
        ]

        run(cmd_video)

    print("\n✅ Bundle finalizado com sucesso.")


if __name__ == "__main__":
    main()
