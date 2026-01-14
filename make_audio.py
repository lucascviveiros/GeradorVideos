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
- Opcional: segmentos por parágrafo + concat em MP3

Requisitos:
  pip install TTS==0.22.0 soundfile
  ffmpeg no PATH (para converter/concatenar MP3)

Docs/refs:
- Coqui TTS + XTTS: https://docs.coqui.ai/en/latest/models/xtts.html
"""

from __future__ import annotations
import sys
import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
    paras = [p.strip() for p in re.split(r"\n\s*\n+", raw) if p.strip()]
    return paras


def strip_line_final_punct(s: str, lang: str) -> str:
    """
    Normaliza fim de frase para TTS.

    - PT: remove .!? para evitar o modelo falar "ponto", mas preserva pausa com "\n\n".
    - EN/ES: mantém .!? (ajuda prosódia) e insere "\n\n" entre frases (respiração).
    - Preserva decimais (3.10).
    """
    s = s.strip()

    if lang == "pt":
        # PT: converter fim de frase em respiração SEM manter a pontuação
        s = re.sub(
            r"(?<!\d)([.!?]+)(\s+)(?=[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ])",
            r"\n\n",
            s
        )
        # remove pontuação final se sobrar no fim
        s = re.sub(r"(?<!\d)[.!?]+\s*$", "", s)

    else:
        # EN/ES: manter pontuação, mas inserir respiração entre frases
        s = re.sub(
            r"(?<!\d)([.!?])(\s+)(?=[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ])",
            r"\1\n\n",
            s
        )

    # normaliza respirações repetidas
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()




def normalize_text_for_tts(s: str) -> str:
    # Normalização leve para evitar artefatos
    s = s.strip()

    # Remove aspas (retas e tipográficas) para o TTS não "pegar" esses caracteres
    # Inclui: "  '  “ ”  ‘ ’  „  « »  `
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
    sample_rate: Optional[int],
) -> None:
    """
    Gera áudio e grava WAV PCM_16.
    - Se sample_rate vier None, tenta derivar do modelo (mais correto).
    - Se não conseguir, cai em 24000.
    """
    ensure_soundfile()
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    wav = tts.tts(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
    )

    sr = sample_rate
    if sr is None:
        # tenta puxar do synthesizer se existir
        sr = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", None)
    if not sr:
        sr = 24000

    sf.write(str(wav_path), wav, int(sr), subtype="PCM_16")



def ffmpeg_normalize_wav(
    in_wav: Path,
    out_wav: Path,
    sample_rate: int,
    channels: int = 1,
) -> None:
    """
    Normaliza WAV para PCM16, SR fixo e canais fixos.
    Útil para garantir concat consistente em segmentos.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y",
        "-i", str(in_wav),
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(out_wav),
    ])


def ffmpeg_wav_to_mp3(wav: Path, out_mp3: Path, bitrate: str) -> None:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y",
        "-i", str(wav),
        "-c:a", "libmp3lame",
        "-b:a", bitrate,
        str(out_mp3),
    ])

def get_voice_wav_for_lang(voices_dir: Path, lang: str, wav_sr: int) -> Path:
    """
    Retorna um voice_{lang}.wav.
    - Se existir WAV, usa.
    - Se só existir MP3, converte para WAV (PCM16 mono SR fixo) e passa a usar o WAV.
    """
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


def ffmpeg_concat_wavs_to_mp3(wavs: List[Path], out_mp3: Path, bitrate: str) -> None:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)

    def esc(p: Path) -> str:
        # concat demuxer aceita: file "path"
        s = p.as_posix()
        return s.replace('"', r'\"')

    list_file = out_mp3.with_suffix(".concat.txt")
    list_file.write_text("\n".join([f'file "{esc(w)}"' for w in wavs]), encoding="utf-8")

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

    # NOVO: estabilidade de áudio (SR fixo + normalização opcional)
    ap.add_argument("--wav_sr", type=int, default=24000,
                    help="Sample rate do WAV gerado (default: 24000; recomendado para XTTS v2).")
    ap.add_argument("--normalize_wavs", action="store_true",
                    help="Normaliza WAVs (PCM16, mono, SR fixo) antes de concatenar/converter.")

    # Compat: aceita --text, mas não é mais necessário
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

    # Monta specs por idioma (TXT e voice obrigatórios)
    specs: List[LangSpec] = []
    for lang in langs:
        txt = require_file(narr_dir / f"{ep}_{lang}.txt", label=f"texto {lang}")
        voice = require_file(voices_dir / f"voice_{lang}.wav", label=f"voz {lang}")
        specs.append(LangSpec(lang=lang, text_path=txt, voice_path=voice))

    def split_big(p: str, max_chars: int) -> List[str]:
        """
        Divide texto grande em blocos <= max_chars com cortes naturais:
        1) tenta dividir por fim de frase (. ! ? …)
        2) agrupa sentenças até max_chars
        3) se alguma sentença for maior que max_chars, divide por palavras
        """
        p = p.strip()
        if len(p) <= max_chars:
            return [p]

        # 1) split por fim de frase mantendo pontuação
        sentences = re.split(r"(?<=[.!?…])\s+", p)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: List[str] = []
        cur = ""

        def flush():
            nonlocal cur
            if cur:
                chunks.append(cur.strip())
                cur = ""

        # 2) agrupa sentenças
        for s in sentences:
            if len(s) <= max_chars:
                if len(cur) + len(s) + (1 if cur else 0) <= max_chars:
                    cur = (cur + " " + s).strip()
                else:
                    flush()
                    cur = s
            else:
                # 3) sentença enorme: quebra por palavras
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


    # Carrega modelo uma vez
    tts = TTS(MODEL_NAME)
    tts = tts.to("cuda" if args.gpu else "cpu")

    for spec in specs:
        lang = spec.lang
        paras = [strip_line_final_punct(normalize_text_for_tts(p)) for p in read_paragraphs(spec.text_path)]
        if not paras:
            raise RuntimeError(f"Nenhum parágrafo encontrado em: {spec.text_path}")

        out_mp3 = out_dir / f"{ep}_{lang}.mp3"

        if args.segments:
            seg_wavs: List[Path] = []
            seg_idx = 1

            for p in paras:
                for piece in split_big(p, args.max_chars):
                    raw_wav = out_dir / f"{ep}_{lang}_{seg_idx:02d}.raw.wav"
                    final_wav = out_dir / f"{ep}_{lang}_{seg_idx:02d}.wav"
                    piece = strip_line_final_punct(piece)

                    tts_piece_to_wav(
                        tts=tts,
                        text=piece,
                        speaker_wav=str(spec.voice_path),
                        language=lang,
                        wav_path=raw_wav,
                        sample_rate=args.wav_sr,
                    )

                    if args.normalize_wavs:
                        ffmpeg_normalize_wav(raw_wav, final_wav, sample_rate=args.wav_sr, channels=1)
                        try:
                            raw_wav.unlink(missing_ok=True)
                        except Exception:
                            pass
                        seg_wavs.append(final_wav)
                    else:
                        seg_wavs.append(raw_wav)

                    seg_idx += 1

            ffmpeg_concat_wavs_to_mp3(seg_wavs, out_mp3, bitrate=args.mp3_bitrate)
            print(f"OK ({lang}) segments+mp3: {out_mp3}")

        else:
            tmp_raw = out_dir / f"{ep}_{lang}.raw.wav"
            tmp_wav = out_dir / f"{ep}_{lang}.wav"
            full_text = "\n\n".join(paras)
            full_text = strip_line_final_punct(full_text)

            tts_piece_to_wav(
                tts=tts,
                text=full_text,
                speaker_wav=str(spec.voice_path),
                language=lang,
                wav_path=tmp_raw,
                sample_rate=args.wav_sr,
            )

            if args.normalize_wavs:
                ffmpeg_normalize_wav(tmp_raw, tmp_wav, sample_rate=args.wav_sr, channels=1)
                try:
                    tmp_raw.unlink(missing_ok=True)
                except Exception:
                    pass
                source_wav = tmp_wav
            else:
                source_wav = tmp_raw

            ffmpeg_wav_to_mp3(source_wav, out_mp3, bitrate=args.mp3_bitrate)

            try:
                source_wav.unlink(missing_ok=True)
            except Exception:
                pass

            print(f"OK ({lang}) mp3: {out_mp3}")


if __name__ == "__main__":
    main()
