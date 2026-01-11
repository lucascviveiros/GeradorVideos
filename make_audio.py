#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Exemplo:
# py -3.10 .\make_audio.py --episode ep001 --narr_dir narrativa --out_dir audio --langs pt,en,es --gpu --voices_dir voices
#
# (Opcional) ainda aceita --text como compat, mas o modo recomendado é acima.

"""
Coqui XTTS v2 (local) - Batch TTS PT/EN/ES
- Um TXT por idioma: narrativa/ep001_pt.txt, ep001_en.txt, ep001_es.txt
- Uma voz por idioma (obrigatório): voices/voice_pt.mp3, voice_en.mp3, voice_es.mp3
- GPU suportada (RTX)
- Saída: audio/ep001_pt.mp3, audio/ep001_en.mp3, audio/ep001_es.mp3
- Opcional: segmentos por parágrafo + concat em MP3

Requisitos:
  pip install TTS==0.22.0
  ffmpeg no PATH (para converter/concatenar MP3)

Docs/refs:
- Coqui TTS + XTTS: https://docs.coqui.ai/en/latest/models/xtts.html
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from TTS.api import TTS

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


# -------------------------
# Utils
# -------------------------

def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n\n{p.stdout}")


def ensure_ffmpeg() -> None:
    try:
        _run(["ffmpeg", "-version"])
    except Exception as e:
        raise RuntimeError("ffmpeg não encontrado no PATH. Instale e adicione ao PATH.") from e


def read_paragraphs(txt_path: Path) -> List[str]:
    raw = txt_path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    # Parágrafo = bloco separado por linha em branco
    paras = [p.strip() for p in re.split(r"\n\s*\n+", raw) if p.strip()]
    return paras


def normalize_text_for_tts(s: str) -> str:
    # Normalização leve para evitar artefatos
    s = s.strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def ffmpeg_wav_to_mp3(wav: Path, out_mp3: Path, bitrate: str) -> None:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y",
        "-i", str(wav),
        "-c:a", "libmp3lame",
        "-b:a", bitrate,
        str(out_mp3),
    ])


def ffmpeg_concat_wavs_to_mp3(wavs: List[Path], out_mp3: Path, bitrate: str) -> None:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)

    # Concat demuxer (arquivo lista)
    list_file = out_mp3.with_suffix(".concat.txt")
    list_file.write_text("\n".join([f"file '{w.as_posix()}'" for w in wavs]), encoding="utf-8")

    _run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c:a", "libmp3lame",
        "-b:a", bitrate,
        str(out_mp3),
    ])

    try:
        list_file.unlink(missing_ok=True)
    except Exception:
        pass


def require_file(p: Path, label: str) -> Path:
    try:
        return p.expanduser().resolve(strict=True)
    except Exception as e:
        raise RuntimeError(f"Arquivo obrigatório não encontrado ({label}): {p}") from e


@dataclass
class LangSpec:
    lang: str
    text_path: Path
    voice_path: Path


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Coqui XTTS v2 local: gera MP3 PT/EN/ES usando 1 TXT por idioma e 1 voice mp3 por idioma."
    )

    # NOVO: episódio + diretórios
    ap.add_argument("--episode", required=True, help="Ex: ep001 (vai buscar ep001_pt.txt, ep001_en.txt, ep001_es.txt)")
    ap.add_argument("--narr_dir", default="narrativa", help="Pasta onde ficam os .txt por idioma (default: narrativa)")
    ap.add_argument("--voices_dir", default="voices", help="Pasta onde ficam voice_pt.mp3 etc. (default: voices)")
    ap.add_argument("--out_dir", default="audio", help="Pasta de saída (default: audio)")

    ap.add_argument("--langs", default="pt,en,es", help="Idiomas: pt,en,es")
    ap.add_argument("--gpu", action="store_true", help="Usar GPU (recomendado).")
    ap.add_argument("--mp3_bitrate", default="192k", help="Bitrate MP3 (default: 192k)")

    # Mantido: segmentos
    ap.add_argument("--segments", action="store_true",
                    help="Gera WAV por parágrafo (ep001_pt_01.wav...) e concatena em MP3.")
    ap.add_argument("--max_chars", type=int, default=650,
                    help="Se um parágrafo for muito longo, divide em blocos <= max_chars (default: 650).")

    # Compat: aceita --text, mas não é mais necessário
    ap.add_argument("--text", default=None, help="(Compat) Ignorado no modo por-idioma; use --episode + --narr_dir.")

    args = ap.parse_args()

    ensure_ffmpeg()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    narr_dir = Path(args.narr_dir).expanduser().resolve()
    voices_dir = Path(args.voices_dir).expanduser().resolve()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    if not langs:
        raise RuntimeError("--langs inválido. Ex: pt,en,es")

    for lang in langs:
        if lang not in ("pt", "en", "es"):
            raise RuntimeError(f"Idioma não suportado neste script: {lang}. Use pt,en,es.")

    ep = args.episode.strip()
    if not ep:
        raise RuntimeError("--episode inválido.")

    # Monta specs por idioma (TXT e voice obrigatórios)
    specs: List[LangSpec] = []
    for lang in langs:
        txt = require_file(narr_dir / f"{ep}_{lang}.txt", label=f"texto {lang}")
        voice = require_file(voices_dir / f"voice_{lang}.mp3", label=f"voz {lang}")
        specs.append(LangSpec(lang=lang, text_path=txt, voice_path=voice))

    def split_big(p: str, max_chars: int) -> List[str]:
        if len(p) <= max_chars:
            return [p]
        chunks: List[str] = []
        cur = ""
        for line in p.split("\n"):
            line = line.strip()
            if not line:
                continue
            if len(cur) + len(line) + 1 <= max_chars:
                cur = (cur + " " + line).strip()
            else:
                if cur:
                    chunks.append(cur)
                cur = line
        if cur:
            chunks.append(cur)
        return chunks

    # Carrega modelo uma vez
    tts = TTS(MODEL_NAME)
    tts = tts.to("cuda" if args.gpu else "cpu")

    for spec in specs:
        lang = spec.lang
        paras = [normalize_text_for_tts(p) for p in read_paragraphs(spec.text_path)]
        if not paras:
            raise RuntimeError(f"Nenhum parágrafo encontrado em: {spec.text_path}")

        out_mp3 = out_dir / f"{ep}_{lang}.mp3"

        if args.segments:
            seg_wavs: List[Path] = []
            seg_idx = 1

            for p in paras:
                for piece in split_big(p, args.max_chars):
                    wav_path = out_dir / f"{ep}_{lang}_{seg_idx:02d}.wav"
                    tts.tts_to_file(
                        text=piece,
                        speaker_wav=str(spec.voice_path),
                        language=lang,
                        file_path=str(wav_path),
                    )
                    seg_wavs.append(wav_path)
                    seg_idx += 1

            ffmpeg_concat_wavs_to_mp3(seg_wavs, out_mp3, bitrate=args.mp3_bitrate)
            print(f"OK ({lang}) segments+mp3: {out_mp3}")

        else:
            tmp_wav = out_dir / f"{ep}_{lang}.wav"
            full_text = "\n\n".join(paras)

            tts.tts_to_file(
                text=full_text,
                speaker_wav=str(spec.voice_path),
                language=lang,
                file_path=str(tmp_wav),
            )

            ffmpeg_wav_to_mp3(tmp_wav, out_mp3, bitrate=args.mp3_bitrate)
            try:
                tmp_wav.unlink(missing_ok=True)
            except Exception:
                pass

            print(f"OK ({lang}) mp3: {out_mp3}")


if __name__ == "__main__":
    main()
