#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Exemplo:
# py -3.10 .\make_audio.py --episode ep001 --narr_dir narrativa --out_dir audio --langs pt,en,es --gpu --voices_dir voices
#
# (Opcional) ainda aceita --text como compat, mas o modo recomendado é acima.

# convertendo para wav
#ffmpeg -hide_banner -loglevel error -y -i "X:\GeradorVideos\voices\voice_pt.mp3" -vn -ac 1 -ar 24000 -c:a pcm_s16le "X:\GeradorVideos\voices\voice_pt.wav"
#ffmpeg -hide_banner -loglevel error -y -i "X:\GeradorVideos\voices\voice_en.mp3" -vn -ac 1 -ar 24000 -c:a pcm_s16le "X:\GeradorVideos\voices\voice_en.wav"
#ffmpeg -hide_banner -loglevel error -y -i "X:\GeradorVideos\voices\voice_es.mp3" -vn -ac 1 -ar 24000 -c:a pcm_s16le "X:\GeradorVideos\voices\voice_es.wav"


"""
Coqui XTTS v2 (local) - Batch TTS PT/EN/ES
- Um TXT por idioma: narrativa/ep001_pt.txt, ep001_en.txt, ep001_es.txt
- Uma voz por idioma (obrigatório): voices/voice_pt.mp3, voice_en.mp3, voice_es.mp3
- GPU suportada (RTX)
- Saída: audio/ep001_pt.mp3, audio/ep001_en.mp3, audio/ep001_es.mp3
- Opcional: segmentos por chunk + concat em WAV mestre + encode MP3 (normal e fast)

Requisitos:
  pip install TTS==0.22.0 soundfile
  ffmpeg no PATH (para converter/concatenar/encodar)

Docs/refs:
- Coqui TTS + XTTS: https://docs.coqui.ai/en/latest/models/xtts.html
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from TTS.api import TTS

try:
    import soundfile as sf
except Exception:
    sf = None


MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


# -------------------------
# Utils
# -------------------------

def ensure_soundfile() -> None:
    if sf is None:
        raise RuntimeError(
            "Dependência ausente: 'soundfile'.\n"
            "Instale com: pip install soundfile\n"
            "Obs: no Windows, o wheel normalmente já inclui o necessário."
        )


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
    return [p.strip() for p in re.split(r"\n\s*\n+", raw) if p.strip()]


def detect_silence_edges(samples, thr: float = 0.002, pad: int = 0):
    # fallback simples caso numpy não exista.
    try:
        import numpy as np
        x = np.asarray(samples)
        ax = np.abs(x)
        n = len(ax)

        i0 = 0
        while i0 < n and ax[i0] < thr:
            i0 += 1

        i1 = n - 1
        while i1 > 0 and ax[i1] < thr:
            i1 -= 1

        i0 = max(i0 - pad, 0)
        i1 = min(i1 + pad, n - 1)
        return i0, i1
    except Exception:
        n = len(samples)
        i0 = 0
        while i0 < n and abs(float(samples[i0])) < thr:
            i0 += 1

        i1 = n - 1
        while i1 > 0 and abs(float(samples[i1])) < thr:
            i1 -= 1

        i0 = max(i0 - pad, 0)
        i1 = min(i1 + pad, n - 1)
        return i0, i1


def wav_duration_seconds(wav_path: Path) -> float:
    ensure_soundfile()
    info = sf.info(str(wav_path))
    return float(info.frames) / float(info.samplerate)


def strip_line_final_punct(s: str, lang: str) -> str:
    """
    Normaliza fim de frase para TTS.

    - PT: remove .!? para evitar o modelo falar "ponto", mas preserva pausa com "\n\n".
    - EN/ES: mantém .!? (ajuda prosódia) e insere "\n\n" entre frases (respiração).
    - Preserva decimais (3.10).
    """
    s = s.strip()

    if lang == "pt":
        s = re.sub(
            r"(?<!\d)([.!?]+)(\s+)(?=[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ])",
            r"\n\n",
            s
        )
        s = re.sub(r"(?<!\d)[.!?]+\s*$", "", s)
    else:
        s = re.sub(
            r"(?<!\d)([.!?])(\s+)(?=[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ])",
            r"\1\n\n",
            s
        )

    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def normalize_text_for_tts(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[\"'“”‘’„«»`]", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def tts_piece_to_wav(
    tts: TTS,
    text: str,
    speaker_wav: str,
    language: str,
    wav_path: Path,
    sample_rate: int,
    debug_silence: bool = False,
) -> None:
    ensure_soundfile()
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    wav = tts.tts(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
    )

    if debug_silence:
        try:
            i0, i1 = detect_silence_edges(wav, thr=0.002, pad=0)
            lead = i0 / float(sample_rate)
            tail = (len(wav) - 1 - i1) / float(sample_rate)
            print(f"    [silence] lead={lead:.3f}s tail={tail:.3f}s len={len(wav)/float(sample_rate):.2f}s")
        except Exception:
            pass

    sf.write(str(wav_path), wav, int(sample_rate), subtype="PCM_16")


def ffmpeg_normalize_wav(
    in_wav: Path,
    out_wav: Path,
    sample_rate: int,
    channels: int = 1,
) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y",
        "-i", str(in_wav),
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(out_wav),
    ])


def ffmpeg_concat_wavs_to_master_wav(wavs: List[Path], out_wav: Path, sample_rate: int) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    def to_ffmpeg_path(p: Path) -> str:
        return str(p.expanduser().resolve()).replace("\\", "/")

    def make_line(p: Path) -> str:
        s = to_ffmpeg_path(p)
        if " " in s or "'" in s:
            s = s.replace("'", r"\'")
            return f"file '{s}'"
        return f"file {s}"

    list_file = out_wav.with_suffix(".concat.txt")
    list_file.write_text("\n".join(make_line(w) for w in wavs) + "\n", encoding="utf-8")

    _run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(out_wav),
    ])

    try:
        list_file.unlink(missing_ok=True)
    except Exception:
        pass


def ffmpeg_encode_mp3_from_wav(
    in_wav: Path,
    out_mp3: Path,
    bitrate: str,
    vbr_q: Optional[int],
    speed: float,
) -> None:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)

    def atempo_filter(speed_val: float) -> Optional[str]:
        if abs(speed_val - 1.0) < 1e-6:
            return None

        parts: List[str] = []
        s = speed_val
        while s > 2.0:
            parts.append("atempo=2.0")
            s /= 2.0
        while s < 0.5:
            parts.append("atempo=0.5")
            s /= 0.5
        parts.append(f"atempo={s:.6f}".rstrip("0").rstrip("."))
        return ",".join(parts)

    cmd = ["ffmpeg", "-y", "-i", str(in_wav)]
    filt = atempo_filter(speed)
    if filt:
        cmd += ["-filter:a", filt]

    cmd += ["-c:a", "libmp3lame"]
    if vbr_q is not None:
        cmd += ["-q:a", str(vbr_q)]
    else:
        cmd += ["-b:a", bitrate]

    cmd += [str(out_mp3)]
    _run(cmd)


def get_voice_wav_for_lang(voices_dir: Path, lang: str, wav_sr: int) -> Path:
    wav = voices_dir / f"voice_{lang}.wav"
    mp3 = voices_dir / f"voice_{lang}.mp3"

    if wav.exists():
        return wav.expanduser().resolve()

    if mp3.exists():
        ensure_ffmpeg()
        out = wav
        out.parent.mkdir(parents=True, exist_ok=True)
        _run([
            "ffmpeg", "-y",
            "-i", str(mp3),
            "-vn",
            "-ac", "1",
            "-ar", str(wav_sr),
            "-c:a", "pcm_s16le",
            str(out),
        ])
        return out.expanduser().resolve()

    raise RuntimeError(
        f"Voz não encontrada para '{lang}'. Esperado: {wav.name} ou {mp3.name} em {voices_dir}"
    )


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


def split_big(p: str, max_chars: int) -> List[str]:
    p = p.strip()
    if len(p) <= max_chars:
        return [p]

    sentences = re.split(r"(?<=[.!?…])\s+", p)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: List[str] = []
    cur = ""

    def flush():
        nonlocal cur
        if cur:
            chunks.append(cur.strip())
            cur = ""

    for s in sentences:
        if len(s) <= max_chars:
            if len(cur) + len(s) + (1 if cur else 0) <= max_chars:
                cur = (cur + " " + s).strip()
            else:
                flush()
                cur = s
        else:
            flush()
            words = s.split()
            wcur = ""
            for w in words:
                if len(wcur) + len(w) + (1 if wcur else 0) <= max_chars:
                    wcur = (wcur + " " + w).strip()
                else:
                    if wcur:
                        chunks.append(wcur)
                    wcur = w
            if wcur:
                chunks.append(wcur)

    flush()
    return chunks


def merge_small_chunks(chunks: List[str], min_chars: int) -> List[str]:
    if min_chars <= 0:
        return chunks

    out: List[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            out.append(buf.strip())
        buf = ""

    for c in chunks:
        c = c.strip()
        if not c:
            continue

        if not buf:
            buf = c
            continue

        if len(buf) < min_chars:
            buf = buf + "\n\n" + c
        else:
            flush()
            buf = c

    flush()
    return out


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Coqui XTTS v2 local: gera MP3 PT/EN/ES usando 1 TXT por idioma e 1 voice por idioma."
    )

    ap.add_argument("--episode", required=True, help="Ex: ep001 (busca ep001_pt.txt, ep001_en.txt, ep001_es.txt)")
    ap.add_argument("--narr_dir", default="narrativa", help="Pasta onde ficam os .txt (default: narrativa)")
    ap.add_argument("--voices_dir", default="voices", help="Pasta onde ficam voice_pt.mp3 etc. (default: voices)")
    ap.add_argument("--out_dir", default="audio", help="Pasta de saída (default: audio)")

    ap.add_argument("--langs", default="pt,en,es", help="Idiomas: pt,en,es")
    ap.add_argument("--gpu", action="store_true", help="Usar GPU (recomendado).")

    ap.add_argument("--segments", action="store_true",
                    help="Gera WAV por chunk e concatena em WAV mestre antes de encodar MP3.")

    ap.add_argument("--max_chars", type=int, default=420,
                    help="Divide chunks longos em <= max_chars (default: 420).")
    ap.add_argument("--min_chars", type=int, default=140,
                    help="Junta chunks curtos até >= min_chars (default: 140).")

    ap.add_argument("--wav_sr", type=int, default=24000,
                    help="Sample rate do WAV (default: 24000; recomendado para XTTS v2).")
    ap.add_argument("--normalize_wavs", action="store_true",
                    help="Normaliza WAVs (PCM16, mono, SR fixo) antes de concatenar.")

    # MP3
    ap.add_argument("--mp3_bitrate", default="192k", help="Bitrate MP3 CBR (default: 192k)")
    ap.add_argument("--mp3_vbr_q", type=int, default=None,
                    help="Se definido, usa VBR (-q:a). Ex: 2 (muito bom).")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Velocidade final (ffmpeg atempo). Ex: 1.20 (default: 1.0)")
    ap.add_argument("--speed_suffix", default="_fast",
                    help="Sufixo do arquivo acelerado (default: _fast).")

    # Debug
    ap.add_argument("--debug_silence", action="store_true", help="Imprime silêncio inicial/final por chunk.")
    ap.add_argument("--debug_metrics", action="store_true", help="Imprime métricas por chunk (dur/cps).")

    # Compat
    ap.add_argument("--text", default=None, help="(Compat) Ignorado no modo por-idioma; use --episode + --narr_dir.")

    args = ap.parse_args()

    ensure_ffmpeg()
    ensure_soundfile()

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

    specs: List[LangSpec] = []
    for lang in langs:
        txt = require_file(narr_dir / f"{ep}_{lang}.txt", label=f"texto {lang}")
        voice = get_voice_wav_for_lang(voices_dir, lang, args.wav_sr)
        specs.append(LangSpec(lang=lang, text_path=txt, voice_path=voice))

    # Carrega modelo uma vez
    tts = TTS(MODEL_NAME)
    tts = tts.to("cuda" if args.gpu else "cpu")

    for spec in specs:
        lang = spec.lang

        paras = [normalize_text_for_tts(p) for p in read_paragraphs(spec.text_path)]
        if not paras:
            raise RuntimeError(f"Nenhum parágrafo encontrado em: {spec.text_path}")

        chunks: List[str] = []
        for p in paras:
            chunks.extend(split_big(p, args.max_chars))

        chunks = [strip_line_final_punct(c, lang) for c in chunks if c.strip()]
        chunks = merge_small_chunks(chunks, args.min_chars)

        if not chunks:
            raise RuntimeError(f"Nenhum chunk após split/merge em: {spec.text_path}")

        out_mp3 = out_dir / f"{ep}_{lang}.mp3"
        out_mp3_fast = out_dir / f"{ep}_{lang}{args.speed_suffix}.mp3"
        master_wav = out_dir / f"{ep}_{lang}.master.wav"

        if args.segments:
            seg_wavs: List[Path] = []

            for i, piece in enumerate(chunks, start=1):
                raw_wav = out_dir / f"{ep}_{lang}_{i:02d}.raw.wav"
                final_wav = out_dir / f"{ep}_{lang}_{i:02d}.wav"

                tts_piece_to_wav(
                    tts=tts,
                    text=piece,
                    speaker_wav=str(spec.voice_path),
                    language=lang,
                    wav_path=raw_wav,
                    sample_rate=args.wav_sr,
                    debug_silence=args.debug_silence,
                )

                if args.normalize_wavs:
                    ffmpeg_normalize_wav(raw_wav, final_wav, sample_rate=args.wav_sr, channels=1)
                    try:
                        raw_wav.unlink(missing_ok=True)
                    except Exception:
                        pass
                    seg_path = final_wav
                else:
                    seg_path = raw_wav

                seg_wavs.append(seg_path)

                if args.debug_metrics:
                    try:
                        dur = wav_duration_seconds(seg_path)
                        text_len = len(piece)
                        word_count = len(piece.split())
                        sec_per_word = dur / max(word_count, 1)
                        cps = text_len / max(dur, 1e-6)
                        print(
                            f"[SEG {i:02d}] dur={dur:5.2f}s words={word_count:3d} "
                            f"sec/word={sec_per_word:4.2f} cps={cps:5.1f} "
                            f"text='{piece[:60]}...'"
                        )
                    except Exception:
                        pass

            ffmpeg_concat_wavs_to_master_wav(seg_wavs, master_wav, sample_rate=args.wav_sr)

        else:
            full_text = "\n\n".join(chunks)
            tts_piece_to_wav(
                tts=tts,
                text=full_text,
                speaker_wav=str(spec.voice_path),
                language=lang,
                wav_path=master_wav,
                sample_rate=args.wav_sr,
                debug_silence=args.debug_silence,
            )

            if args.normalize_wavs:
                tmp = out_dir / f"{ep}_{lang}.master.norm.wav"
                ffmpeg_normalize_wav(master_wav, tmp, sample_rate=args.wav_sr, channels=1)
                try:
                    master_wav.unlink(missing_ok=True)
                except Exception:
                    pass
                master_wav = tmp

        # MP3 normal (sem speed)
        ffmpeg_encode_mp3_from_wav(
            in_wav=master_wav,
            out_mp3=out_mp3,
            bitrate=args.mp3_bitrate,
            vbr_q=args.mp3_vbr_q,
            speed=1.0,
        )
        print(f"OK ({lang}) mp3: {out_mp3}")

        # MP3 fast (se speed != 1.0)
        if abs(float(args.speed) - 1.0) > 1e-6:
            ffmpeg_encode_mp3_from_wav(
                in_wav=master_wav,
                out_mp3=out_mp3_fast,
                bitrate=args.mp3_bitrate,
                vbr_q=args.mp3_vbr_q if args.mp3_vbr_q is not None else 2,
                speed=float(args.speed),
            )
            print(f"OK ({lang}) mp3 fast ({args.speed}x): {out_mp3_fast}")

        # Limpa master wav
        try:
            Path(master_wav).unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
