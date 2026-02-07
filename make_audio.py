#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Exemplo:
# normal: py -3.10 .\make_audio.py --episode ep001 --narr_dir narrativa --out_dir audio --langs pt,en,es --gpu --voices_dir voices
# FAST (New Style):
# py -3.10 .\make_audio.py --episode ep001 --narr_dir narrativa --out_dir audio --langs pt --gpu --voices_dir voices --segments --normalize_wavs --max_chars 420 --min_chars 120 --mp3_vbr_q 2 --speed 1.13 --temperature 0.75 --repetition_penalty 2.5 --trim_silence

"""
Coqui XTTS v2 (local) - Batch TTS PT/EN/ES
- Um TXT por idioma: narrativa/ep001_pt.txt, ep001_en.txt, ep001_es.txt
- Uma voz por idioma (obrigatório): voices/voice_pt.mp3, voice_en.mp3, voice_es.mp3
- GPU suportada (RTX)
- Saída: audio/ep001_pt.mp3, audio/ep001_en.mp3, audio/ep001_es.mp3
- Opcional: segmentos por chunk + concat em WAV mestre + encode MP3 (normal e fast)

Requisitos:
  pip install TTS==0.22.0 soundfile numpy
  ffmpeg no PATH (para converter/concatenar/encodar)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import csv
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict

from TTS.api import TTS

try:
    import soundfile as sf
    import numpy as np
    import torch
    import gc
except Exception:
    sf = None
    np = None
    torch = None
    gc = None

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
FFMPEG_EXE = "ffmpeg"

def reset_xtts_model(tts, use_gpu):
    """Resets the TTS model to clear potential inference hangs."""
    print("    [RESET] Reloading model to fix inference hang...")
    del tts
    if gc: gc.collect()
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    tts = TTS(MODEL_NAME)
    tts = tts.to("cuda" if use_gpu else "cpu")
    return tts

@dataclass
class AudioMetrics:
    filename: str
    text_len: int
    duration_s: float
    rms_dbfs: float
    peak_db: float
    crest_factor: float
    silence_pct: float
    internal_silence_pct: float
    silence_start_s: float
    silence_end_s: float
    zcr_avg: float
    f0_mean: float
    f0_std: float
    chars_per_sec: float
    words_per_sec: float
    flags: str = ""

def estimate_f0_autocorr(x, fs):
    # Very basic autocorrelation pitch estimator
    # Downsample for speed? No, keep it simple.
    if len(x) < fs * 0.05: return 0.0, 0.0 # too short
    
    # Windowing
    w = np.hanning(len(x))
    xw = x * w
    
    # Autocorr
    r = np.correlate(xw, xw, mode='full')
    r = r[len(r)//2:]
    
    # Find first peak after zero
    # min pitch 50Hz -> max lag fs/50
    # max pitch 500Hz -> min lag fs/500
    d_min = int(fs / 500)
    d_max = int(fs / 50)
    
    if len(r) < d_max: return 0.0, 0.0

    valid_r = r[d_min:d_max]
    if len(valid_r) == 0: return 0.0, 0.0
    
    lag = d_min + np.argmax(valid_r)
    if r[lag] < 0.1 * r[0]: return 0.0, 0.0 # weak correlation
    
    f0 = fs / lag
    return float(f0), 0.0 # TODO: std dev requires frame-based processing, keeping it simple 0.0

def analyze_audio_chunk(wav_path: Path, text: str, sample_rate: int) -> AudioMetrics:
    ensure_deps()
    try:
        data, fs = sf.read(str(wav_path))
    except:
        return None
        
    if len(data) == 0: return None
    
    # Mono
    if len(data.shape) > 1:
        data = data[:, 0]
        
    dur = len(data) / fs
    
    # 1. Amplitude Stats
    peak = np.max(np.abs(data))
    rms = np.sqrt(np.mean(data**2))
    
    peak_db = 20 * math.log10(peak) if peak > 1e-9 else -99.0
    rms_db = 20 * math.log10(rms) if rms > 1e-9 else -99.0
    crest = peak / rms if rms > 1e-9 else 0.0
    
    # 2. Silence (Threshold -42dB as requested)
    th_db = -42.0
    th_lin = 10**(th_db/20)
    
    # Rolling window 50ms for cleaner silence detection?
    # Simple sample-based is eager. 
    # Let's use simple magnitude check for speed, but maybe average over window?
    # User said "win 50ms".
    win_len = int(0.05 * fs)
    if win_len < 1: win_len = 1
    
    # Square for energy
    # Efficient: pd.Series.rolling? No pandas.
    # Convolve with ones?
    # Let's stick to simple ABS < Threshold for now, maybe filtered.
    # Or just check sample by sample against threshold? 
    # "janela 50ms" implies filtering.
    
    # Fast envelope
    # reshape to chunks of win_len
    pad_len = int(np.ceil(len(data)/win_len)) * win_len - len(data)
    if pad_len > 0:
        data_pad = np.pad(data, (0, pad_len))
    else:
        data_pad = data
        
    # Reshape (n_wins, win_len)
    # Mean abs per window
    wins = data_pad.reshape(-1, win_len)
    win_amp = np.mean(np.abs(wins), axis=1)
    
    is_silence_win = win_amp < th_lin
    sil_pct = np.sum(is_silence_win) / len(is_silence_win) * 100.0
    
    # Start/End Silence (from windows)
    sil_start_sys = 0
    for s in is_silence_win:
        if not s: break
        sil_start_sys += 1
    sil_start_s = sil_start_sys * 0.05
    
    sil_end_sys = 0
    for s in is_silence_win[::-1]:
        if not s: break
        sil_end_sys += 1
    sil_end_s = sil_end_sys * 0.05
    
    # Internal Silence %
    # Total silence duration - start - end
    total_sil_s = np.sum(is_silence_win) * 0.05
    internal_sil_s = max(0.0, total_sil_s - sil_start_s - sil_end_s)
    internal_silence_pct = (internal_sil_s / dur * 100.0) if dur > 0 else 0.0

    # 3. ZCR
    # Zero crossings
    zc = ((data[:-1] * data[1:]) < 0).sum()
    zcr_avg = zc / dur

    # 4. F0
    # Process in chunks of 50ms to get stats?
    # Global estimate is bad for speech. Let's do frame-based.
    frame_len = int(0.05 * fs)
    hop = frame_len // 2
    f0s = []
    
    for i in range(0, len(data) - frame_len, hop):
        frame = data[i:i+frame_len]
        if np.max(np.abs(frame)) < th_lin: continue
        f, _ = estimate_f0_autocorr(frame, fs)
        if f > 50: f0s.append(f)
        
    f0_mean = float(np.mean(f0s)) if f0s else 0.0
    f0_std = float(np.std(f0s)) if f0s else 0.0
    
    # 5. Speed
    n_chars = len(text)
    n_words = len(text.split())
    cps = n_chars / dur if dur > 0 else 0
    wps = n_words / dur if dur > 0 else 0
    
    # Flags
    flags = []
    # Using check_audio_quality logic mostly, keeping flags for backward compat or debug
    if wps > 4.0: flags.append("FAST")
    if wps < 1.2: flags.append("SLOW")
    if internal_silence_pct > 60: flags.append("PAUSE_HEAVY")
    if crest > 10: flags.append("IMPULSIVE") 
    
    return AudioMetrics(
        filename=wav_path.name,
        text_len=n_chars,
        duration_s=dur,
        rms_dbfs=rms_db,
        peak_db=peak_db,
        crest_factor=crest,
        silence_pct=sil_pct,
        internal_silence_pct=internal_silence_pct,
        silence_start_s=sil_start_s,
        silence_end_s=sil_end_s,
        zcr_avg=zcr_avg,
        f0_mean=f0_mean,
        f0_std=f0_std,
        chars_per_sec=cps,
        words_per_sec=wps,
        flags=",".join(flags)
    )

def check_audio_quality(m: AudioMetrics, text: str) -> Tuple[bool, str, str]:
    """
    Quality Gate as per user.
    Returns (Pass:bool, Reason:str, Action:str)
    """
    reasons = []
    
    # NEW: HANG Check
    # if duration_s >= 20 AND (len(text) <= 90 OR wps <= 0.8)
    if m.duration_s >= 20.0 and (len(text) <= 90 or m.words_per_sec <= 0.8):
        return False, "HANG_SUSPECT", "RESET"

    # 1. RMS (< -23)
    if m.rms_dbfs < -23.0:
        reasons.append(f"RMS too low ({m.rms_dbfs:.1f} < -23)")

    # 2. Internal Silence (> 60%)
    if m.internal_silence_pct > 60.0:
        reasons.append(f"Silence too high ({m.internal_silence_pct:.1f}% > 60%)")

    # 3. WPS (1.2 < wps < 4.0)
    # Only meaningful if text is not super short?
    num_words = max(1, len(text.split()))
    if num_words > 2: # Ignore 1-2 word chunks for speed checks
        if m.words_per_sec < 1.2:
            reasons.append(f"Too SLOW ({m.words_per_sec:.1f} wps)")
        elif m.words_per_sec > 4.0:
            reasons.append(f"Too FAST ({m.words_per_sec:.1f} wps)")
    
    # 4. Duration Check (Expected vs Actual)
    # exp = words / 2.7
    # limit = exp * 2.0
    expected_dur = num_words / 2.7
    limit_dur = expected_dur * 2.0
    if m.duration_s > limit_dur and m.duration_s > 4.0: # Only if > 4s to avoid tiny checks
        reasons.append(f"Duration too long ({m.duration_s:.1f}s > {limit_dur:.1f}s)")
        
    if not reasons:
        return True, "OK", "KEEP"
        
    # Heuristic for action
    main_reason = reasons[0]
    return False, "; ".join(reasons), "RETRY"




def save_metrics_csv(metrics: List[AudioMetrics], path: Path):
    if not metrics: return
    
    # Get keys from first item
    keys = list(asdict(metrics[0]).keys())
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for m in metrics:
            writer.writerow(asdict(m))
            
def print_metrics_summary(metrics: List[AudioMetrics]):
    if not metrics: return
    
    ensure_deps() # for numpy
    print("\n--- Audio Quality Summary ---")
    
    def summarize(name, extractor, fmt=".2f"):
        vals = [extractor(m) for m in metrics]
        if not vals: return
        p50 = np.percentile(vals, 50)
        p90 = np.percentile(vals, 90)
        p99 = np.percentile(vals, 99)
        print(f"  {name:12s}: p50={p50:{fmt}} | p90={p90:{fmt}} | p99={p99:{fmt}}")
        
    summarize("RMS (dB)", lambda m: m.rms_dbfs, ".1f")
    summarize("Speed (cps)", lambda m: m.chars_per_sec, ".1f")
    summarize("Silence %", lambda m: m.silence_pct, ".1f")
    summarize("ZCR", lambda m: m.zcr_avg, ".3f")
    
    # Count flags
    flag_counts = {}
    for m in metrics:
        if m.flags:
            for f in m.flags.split(","):
                f = f.strip()
                if f: flag_counts[f] = flag_counts.get(f, 0) + 1
            
    if flag_counts:
        print(f"  Flags Found: {dict(sorted(flag_counts.items(), key=lambda x: -x[1]))}")
    else:
        print("  Flags Found: None (Clean)")
    print("-----------------------------\n")

def analyze_outliers(metrics: List[AudioMetrics], out_txt_path: Path):
    if not metrics: return
    
    print(f"\n--- Outlier Analysis (Top 10) -> {out_txt_path.name} ---")
    
    reports = []
    
    def report_top_k(label, key_func, reverse=True, k=10):
        # Sort
        sorted_m = sorted(metrics, key=key_func, reverse=reverse)
        top_k = sorted_m[:k]
        
        header = f"\n>>> Top {k} {label}"
        print(header)
        reports.append(header)
        
        for i, m in enumerate(top_k, 1):
            val = key_func(m)
            # Find index in original list? We don't have it in AudioMetrics directly unless we add it or infer from filename
            # filename has index e.g. ep_lang_001.wav
            
            line = (
                f"{i}. [{m.filename}] {val:.2f} | "
                f"Dur: {m.duration_s:.2f}s | WPS: {m.words_per_sec:.1f} | "
                f"Sil: {m.silence_start_s:.2f}s/{m.silence_end_s:.2f}s | "
                f"RMS: {m.rms_dbfs:.1f}dB | Flags: {m.flags}"
            )
            print(line)
            reports.append(line)

    # 1. Faster (High WPS)
    report_top_k("Fastest (Words/sec)", lambda m: m.words_per_sec, reverse=True)
    
    # 2. Slower (Low WPS)
    report_top_k("Slowest (Words/sec)", lambda m: m.words_per_sec, reverse=False)
    
    # 3. High Initial Silence
    report_top_k("High Initial Silence (s)", lambda m: m.silence_start_s, reverse=True)
    
    # 4. Low Tail Silence (Tail Cut Risk?)
    # We want smallest non-zero? Or just smallest?
    # reverse=False means ascending
    report_top_k("Lowest Tail Silence (s)", lambda m: m.silence_end_s, reverse=False)

    # 5. Internal Silence %
    report_top_k("Highest Internal Silence %", lambda m: m.silence_pct, reverse=True)
    
    # 6. Dynamics (Crest Factor)
    report_top_k("Highest Crest Factor", lambda m: m.crest_factor, reverse=True)
    
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    out_txt_path.write_text("\n".join(reports), encoding="utf-8")
    print(f"\nSaved detailed outlier report to: {out_txt_path}")
    print("--------------------------------------------------\n")

# -------------------------
# Silence & Trimming
# -------------------------
def ensure_deps() -> None:
    if sf is None or np is None:
        raise RuntimeError(
            "Dependências ausentes: 'soundfile' ou 'numpy'.\n"
            "Instale com: pip install soundfile numpy\n"
        )


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n\n{p.stdout}")


def resolve_ffmpeg_exe() -> str:
    p = shutil.which("ffmpeg")
    if not p:
        raise RuntimeError("ffmpeg não encontrado no PATH. Instale e adicione ao PATH.")
    rp = str(Path(p).resolve())
    if ("Microsoft\\WinGet\\Links" in rp) or ("Microsoft/WinGet/Links" in rp):
        base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
        if base.exists():
            for cand in base.rglob("ffmpeg.exe"):
                try:
                    _run([str(cand), "-version"])
                    return str(cand)
                except Exception:
                    continue
    _run([rp, "-version"])
    return rp


def ensure_ffmpeg() -> None:
    global FFMPEG_EXE
    try:
        FFMPEG_EXE = resolve_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(f"ffmpeg não encontrado ou não executável: {FFMPEG_EXE}") from e


def wav_duration_seconds(wav_path: Path) -> float:
    ensure_deps()
    info = sf.info(str(wav_path))
    return float(info.frames) / float(info.samplerate)


# -------------------------
# Silence & Trimming
# -------------------------
def detect_silence_edges(samples, thr: float = 0.002, pad: int = 0):
    """
    detects silence using a rolling window to avoid cutting zero-crossings or low-freq waves.
    thr: threshold in linear scale (approx 0.002 ~ -54dB)
    """
    x = np.asarray(samples)
    x_abs = np.abs(x)
    n = len(x)
    
    # Window size: 20ms (approx 480 samples at 24k) gives good granularity
    win_size = 480 
    
    # Compute generic envelope (max in window is cheaper/safer than RMS for edges)
    # We'll just scan chunks
    
    # Scan forward
    i0 = 0
    while i0 < n:
        # Check window energy
        end_win = min(i0 + win_size, n)
        chunk_max = np.max(x_abs[i0:end_win]) if i0 < end_win else 0
        if chunk_max > thr:
            break
        i0 += win_size // 2 # overlap step
        
    # Refine i0 (backtrack to find exact sample in the previous window?)
    # Simply taking the window start is conservative (keeps more silence), which is good.
    
    # Scan backward
    i1 = n
    while i1 > 0:
        start_win = max(i1 - win_size, 0)
        chunk_max = np.max(x_abs[start_win:i1]) if start_win < i1 else 0
        if chunk_max > thr:
            break
        i1 -= win_size // 2
        
    # Apply padding
    i0 = max(0, i0 - pad)
    i1 = min(n, i1 + pad)
    
    if i0 >= i1:
        return 0, 0
        
    return i0, i1

def make_silence_wav(duration: float, sample_rate: int, out_path: Path) -> None:
    """Creates a silent WAV file of specific duration."""
    ensure_deps()
    if duration <= 0:
        # Create minimal silence to avoid ffmpeg concat issues with empty files logic if needed, 
        # but better to just skip. 
        # For safety, creating 1 sample is weird. Let's create 10ms.
        num_samples = int(0.01 * sample_rate)
    else:
        num_samples = int(duration * sample_rate)
        
    silence = np.zeros(num_samples, dtype=np.int16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), silence, sample_rate, subtype="PCM_16")

# -------------------------
# Text Processing
# -------------------------

@dataclass
class ProcessingSegment:
    text: str
    pause_after: float = 0.0

def read_smart_paragraphs(txt_path: Path, paragraph_silence: float, sentence_silence: float) -> List[ProcessingSegment]:
    """
    Reads text and splits into logical blocks.
    - \n\n+ -> Paragraph break (paragraph_silence)
    - Single newlines are treated as spaces.
    """
    raw = txt_path.read_text(encoding="utf-8", errors="ignore").replace("\r", "")
    
    # Split by double newlines (paragraphs)
    raw_paras = re.split(r"\n\s*\n+", raw)
    segments: List[ProcessingSegment] = []

    for i, para in enumerate(raw_paras):
        p = para.strip()
        if not p:
            continue
        
        # Determine pause after this paragraph
        # If it's the last paragraph, no pause needed potentially, but consistency is good.
        is_last = (i == len(raw_paras) - 1)
        pause = 0.0 if is_last else paragraph_silence
        
        segments.append(ProcessingSegment(text=p, pause_after=pause))

    return segments

def split_smart(text: str, max_chars: int) -> List[str]:
    """
    Splits text respecting punctuation hierarchy to avoid mid-sentence cuts if possible.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    # Try split by strong punctuation first
    # This regex keeps the separator at the end of the previous chunk
    parts = re.split(r"(?<=[.!?])\s+", text)
    if all(len(p) <= max_chars for p in parts):
        return [p for p in parts if p.strip()]

    # If simple split didn't work, we need to regroup
    final_chunks: List[str] = []
    
    # Helper to accumulate
    current_chunk = ""

    for p in parts:
        p = p.strip()
        if not p: continue

        if len(p) > max_chars:
            # Sub-sentence split needed (commas, then spaces)
            if current_chunk:
                final_chunks.append(current_chunk)
                current_chunk = ""
            
            # Try splitting by comma
            subparts = re.split(r"(?<=,)\s+", p)
            sub_cur = ""
            for sp in subparts:
                sp = sp.strip()
                if not sp: continue
                
                if len(sp) > max_chars:
                     # Emergency split by space
                     words = sp.split()
                     w_cur = ""
                     for w in words:
                         if len(w_cur) + len(w) + 1 <= max_chars:
                             w_cur = (w_cur + " " + w).strip()
                         else:
                             if w_cur: 
                                 if sub_cur: # flush sub_cur wrapper
                                     pass 
                                 final_chunks.append(w_cur)
                             w_cur = w
                     if w_cur:
                        if len(sub_cur) + len(w_cur) + 1 <= max_chars:
                             sub_cur = (sub_cur + " " + w_cur).strip()
                        else:
                             if sub_cur: final_chunks.append(sub_cur)
                             sub_cur = w_cur
                else:
                    if len(sub_cur) + len(sp) + 1 <= max_chars:
                        sub_cur = (sub_cur + " " + sp).strip()
                    else:
                        if sub_cur: final_chunks.append(sub_cur)
                        sub_cur = sp
            if sub_cur:
                final_chunks.append(sub_cur)

        else:
            if len(current_chunk) + len(p) + 1 <= max_chars:
                current_chunk = (current_chunk + " " + p).strip()
            else:
                if current_chunk: final_chunks.append(current_chunk)
                current_chunk = p
    
    if current_chunk:
        final_chunks.append(current_chunk)

    return final_chunks

def process_segments(
    base_segments: List[ProcessingSegment], 
    min_chars: int, 
    max_chars: int, 
    sentence_silence: float
) -> List[ProcessingSegment]:
    """
    Expands paragraphs into XTTS-friendly chunks, merging short ones and preserving silence logic.
    """
    final_list: List[ProcessingSegment] = []

    for seg in base_segments:
        # Split paragraph into manageable chunks
        chunks = split_smart(seg.text, max_chars)
        
        # Merge small chunks logic
        merged_chunks: List[str] = []
        buf = ""

        def flush_buf():
            nonlocal buf
            if buf:
                merged_chunks.append(buf.strip())
                buf = ""

        for c in chunks:
            c = c.strip()
            if not c: continue
            
            if not buf:
                buf = c
            else:
                # If adding this chunk keeps us under max_chars strongly? 
                # OR if buf is too small (< min_chars) we force merge provided it fits max_chars
                
                combined_len = len(buf) + len(c) + 1
                
                if len(buf) < min_chars:
                    # buffer is small, MUST merge if possible
                    if combined_len <= max_chars:
                        buf = buf + " " + c
                    else:
                        # Cannot merge because it would exceed max, so we flush small buf :(
                        flush_buf()
                        buf = c
                else:
                    # buffer is big enough.
                    # Should we merge? Only if it creates a better flow?
                    # Let's say we don't merge unless forced by min_chars of *next* chunk?
                    # Actually logic: chunk stream.
                    
                    # Original logic: if len(buf) < min_chars: buf += c
                    # Here we want to respect min_chars.
                    
                    flush_buf()
                    buf = c
                    
        flush_buf()
        
        # Now create ProcessingSegments
        # If a single paragraph became multiple chunks, we put 'sentence_silence' between them,
        # and 'seg.pause_after' after the *last* chunk of the paragraph.
        
        n = len(merged_chunks)
        for i, chunk_text in enumerate(merged_chunks):
            is_last_sub = (i == n - 1)
            
            # Pause logic:
            # If internal chunk -> sentence_silence
            # If last chunk of paragraph -> original paragraph pause
            p_dur = seg.pause_after if is_last_sub else sentence_silence
            
            final_list.append(ProcessingSegment(text=chunk_text, pause_after=p_dur))

    return final_list

def normalize_text_for_tts(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[\"'“”‘’„«»`]", "", s)
    # Replace all whitespace sequences (including \n) with a single space
    s = re.sub(r"\s+", " ", s)
    return s

def split_text_for_retry(text: str) -> Tuple[str, str]:
    """
    Finds the best split point to divide a chunk into two smaller pieces.
    Prioritizes:
    1. Sentence endings (.?!)
    2. Commas (,)
    3. Spaces
    Returns (part1, part2).
    """
    text = text.strip()
    n = len(text)
    if n < 10: return text, ""

    # Target: closer to middle is better
    mid = n // 2
    
    best_idx = -1
    min_cost = float('inf')

    # Scan all possible split points
    # We look for [CHAR][SPACE]
    # Cost = DistanceFromMid + Penalty(Type)
    
    # Iterate over all characters to find separators followed by space
    for i in range(1, n - 1):
        char = text[i]
        next_char = text[i+1]
        
        # Must act as separator? usually followed by space or EOS
        # But we only split if we stand at a space or after punctuation
        
        is_split = False
        penalty = 1000
        
        if next_char == ' ':
            if char in ".?!":
                penalty = 0 # Best
                is_split = True
            elif char == ',':
                penalty = n * 0.15 # Medium
                is_split = True
            elif char == ' ': # already space?
                # split at i
                penalty = n * 0.4 # Worst
                is_split = True
            # else just a word boundary?
            # actually we split at the space.
        
        if char == ' ':
            # split at space
            penalty = n * 0.4
            is_split = True
            
        if is_split:
            dist = abs(i - mid)
            cost = dist + penalty
            if cost < min_cost:
                min_cost = cost
                best_idx = i + 1 if char in ".?!," else i # include punct in left part
                
                # correction for space split: 
                # if char is space, we want left part up to i, right part from i+1
                if char == ' ': best_idx = i

    if best_idx > 0:
        return text[:best_idx].strip(), text[best_idx:].strip()
        
    return text, ""

# -------------------------
# TTS Core
# -------------------------

def tts_piece_to_wav(
    tts: TTS,
    text: str,
    speaker_wav: str,
    language: str,
    wav_path: Path,
    sample_rate: int,
    # Inference controls
    temperature: float = 0.75,
    repetition_penalty: float = 2.0,
    top_p: float = 0.85,
    top_k: int = 50,
    # Post-proc
    trim_silence: bool = False,
    debug_silence: bool = False,
) -> None:
    ensure_deps()
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean text specifically for XTTS quirks
    # Remove final punctuation if language is PT to avoid "ponto" sometimes, 
    # but XTTS v2 is better than v1. Let's trust normalize_text_for_tts input and maybe just strip trailing dot if short.
    clean_text = normalize_text_for_tts(text)

    # Generate
    try:
        wav = tts.tts(
            text=clean_text,
            speaker_wav=speaker_wav,
            language=language,
            split_sentences=False, 
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
        )
    except Exception as e:
        print(f"!!! Error generating TTS for chunk: '{clean_text[:50]}...': {e}")
        # fallback: generate silence
        wav = np.zeros(int(sample_rate * 0.5), dtype=np.float32).tolist()

    # Process audio (Anti-clip + Trim)
    try:
        x = np.asarray(wav, dtype=np.float32)
        
        # 1. Anti-clipping
        if x.size > 0:
            peak = float(np.max(np.abs(x)))
            target_peak = 0.95
            if peak > 0 and peak > target_peak:
                x *= (target_peak / peak)
        
        # 2. Trim Silence (Smart Crop)
        if trim_silence and x.size > 0:
             # Trim edges
             # Relaxed threshold (0.003) and generous padding (150ms) to avoid cutting tails/breath
             i0, i1 = detect_silence_edges(x, thr=0.003, pad=int(0.15 * sample_rate)) 
             if i0 < i1:
                 x = x[i0:i1]
             else:
                 # is silence? keep a tiny bit to not crash
                 x = np.zeros(int(sample_rate * 0.1), dtype=np.float32)

             if debug_silence:
                lead_s = i0 / sample_rate
                tail_s = (len(wav) - i1) / sample_rate
                print(f"    [Trim] cut {lead_s:.2f}s start, {tail_s:.2f}s end for '{text[:20]}...'")

        wav = x
    except Exception:
        pass

    sf.write(str(wav_path), wav, int(sample_rate), subtype="PCM_16")


def ffmpeg_normalize_wav(in_wav: Path, out_wav: Path, sample_rate: int) -> None:
    _run([
        FFMPEG_EXE, "-y", "-i", str(in_wav),
        "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
        str(out_wav),
    ])

def ffmpeg_concat_wavs_to_master_wav(wavs: List[Path], out_wav: Path, sample_rate: int) -> None:
    """
    Concatenates a list of WAV files.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if empty
    valid_wavs = []
    for w in wavs:
        if w.exists() and w.stat().st_size > 44:
            valid_wavs.append(w)
            
    if not valid_wavs:
        # Create 1s silence if nothing
        make_silence_wav(1.0, sample_rate, out_wav)
        return

    def to_ffmpeg_path(p: Path) -> str:
        return str(p.expanduser().resolve()).replace("\\", "/")

    list_file = out_wav.with_suffix(".concat.txt")
    lines = []
    for w in valid_wavs:
        s = to_ffmpeg_path(w)
        lines.append(f"file '{s}'")
        
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _run([
        FFMPEG_EXE, "-y",
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

    # We use 'atempo' for speed. 
    # Note: XTTS often speaks fast, so speed=1.0 is safer unless user wants fast.
    
    filt_parts: List[str] = []
    
    if abs(speed - 1.0) > 1e-4:
        s = float(speed)
        while s > 2.0:
            filt_parts.append("atempo=2.0")
            s /= 2.0
        while s < 0.5:
             filt_parts.append("atempo=0.5")
             s /= 0.5
        filt_parts.append(f"atempo={s:.5f}")

    # Limiter
    filt_parts.append("alimiter=limit=0.95")

    cmd = [FFMPEG_EXE, "-y", "-i", str(in_wav)]
    if filt_parts:
        cmd += ["-filter:a", ",".join(filt_parts)]

    cmd += ["-c:a", "libmp3lame"]
    if vbr_q is not None:
        cmd += ["-q:a", str(int(vbr_q))]
    else:
        cmd += ["-b:a", str(bitrate)]

    cmd += [str(out_mp3)]
    _run(cmd)


def get_voice_wav_for_lang(voices_dir: Path, lang: str, wav_sr: int) -> Path:
    # Try generic voice first
    wav = voices_dir / f"voice_{lang}.wav"
    mp3 = voices_dir / f"voice_{lang}.mp3"
    
    if wav.exists(): return wav.expanduser().resolve()
    if mp3.exists():
        # Convert to WAV
        ensure_ffmpeg()
        _run([FFMPEG_EXE, "-y", "-i", str(mp3), "-vn", "-ac", "1", "-ar", str(wav_sr), "-c:a", "pcm_s16le", str(wav)])
        return wav.expanduser().resolve()

    raise RuntimeError(f"Voz não encontrada para '{lang}' em {voices_dir}")


def require_file(p: Path, label: str) -> Path:
    try:
        return p.expanduser().resolve(strict=True)
    except Exception as e:
        raise RuntimeError(f"Arquivo obrigatório não encontrado ({label}): {p}") from e

# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Coqui XTTS v2 Batch - ElevenLabs Style Quality")
    
    # Paths
    ap.add_argument("--episode", required=True, help="Ex: ep001")
    ap.add_argument("--narr_dir", default="narrativa", help="Dir TXT input")
    ap.add_argument("--voices_dir", default="voices", help="Dir voices input")
    ap.add_argument("--out_dir", default="audio", help="Dir Output")
    ap.add_argument("--langs", default="pt,en,es", help="pt,en,es")
    
    # TTS / Inference
    ap.add_argument("--gpu", action="store_true", help="Use CUDA")
    ap.add_argument("--temperature", type=float, default=0.65, help="XTTS temp (low = stable, default 0.65)")
    ap.add_argument("--repetition_penalty", type=float, default=1.2, help="XTTS rep penalty (default 1.2)")
    ap.add_argument("--top_p", type=float, default=0.85, help="XTTS top_p")
    ap.add_argument("--top_k", type=int, default=50, help="XTTS top_k")
    
    # Text Processing
    ap.add_argument("--segments", action="store_true", default=True, help="Force segmented mode (Always True basically)")
    ap.add_argument("--max_chars", type=int, default=250, help="Max chars per chunk")
    ap.add_argument("--min_chars", type=int, default=65, help="Min chars to merge")
    
    # Silence / Pacing
    ap.add_argument("--trim_silence", action="store_true", help="Trim start/end silence of chunks")
    ap.add_argument("--paragraph_silence", type=float, default=0.3, help="Silence after paragraph (sec)")
    ap.add_argument("--sentence_silence", type=float, default=0.1, help="Silence after chunk/sentence (sec)")

    # Output Format
    ap.add_argument("--wav_sr", type=int, default=24000)
    ap.add_argument("--normalize_wavs", action="store_true", help="Normalize WAV chunks")
    ap.add_argument("--mp3_bitrate", default="192k")
    ap.add_argument("--mp3_vbr_q", type=int, default=2)
    ap.add_argument("--speed", type=float, default=1.0)
    
    # Debug
    ap.add_argument("--debug_metrics", action="store_true")
    ap.add_argument("--debug_outliers", action="store_true", help="Analyze and print top outlier chunks")
    ap.add_argument("--debug_silence", action="store_true")

    args = ap.parse_args()

    ensure_deps()
    ensure_ffmpeg()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    narr_dir = Path(args.narr_dir).expanduser().resolve()
    voices_dir = Path(args.voices_dir).expanduser().resolve()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    if not langs: raise RuntimeError("Nenhum idioma")

    ep = args.episode.strip()
    
    # Load Model
    print(f"-> Loading model {MODEL_NAME}...")
    tts = TTS(MODEL_NAME)
    tts = tts.to("cuda" if args.gpu else "cpu")
    print("-> Model loaded.")
    
    # Conservative Params for Retry
    CONSERVATIVE_PARAMS = {
        "temperature": 0.50,
        "repetition_penalty": 1.05,
        "top_p": 0.75,
        "top_k": 30
    }

    for lang in langs:
        txt_path = narr_dir / f"{ep}_{lang}.txt"
        if not txt_path.exists():
            print(f"Skipping {lang}, file not found: {txt_path}")
            continue
            
        voice_path = get_voice_wav_for_lang(voices_dir, lang, args.wav_sr)
        
        print(f"\n[{lang.upper()}] Processing {txt_path.name}...")

        # 1. Read & Segment
        base_segments = read_smart_paragraphs(txt_path, args.paragraph_silence, args.sentence_silence)
        
        final_segments = process_segments(
            base_segments, 
            min_chars=args.min_chars, 
            max_chars=args.max_chars, 
            sentence_silence=args.sentence_silence
        )
        
        print(f"    -> {len(final_segments)} chunks ready.")
        
        # 2. Generate with Quality Gate Queue
        seg_wavs_final_list: List[Path] = []
        seg_metrics: List[AudioMetrics] = []
        failed_chunks_log: List[str] = []
        
        # Queue items: dict(seg, id, attempt, params, reset_done)
        queue = []
        for i, seg in enumerate(final_segments, start=1):
            queue.append({
                "seg": seg,
                "id": f"{i:03d}",
                "attempt": 0,
                "params": None,
                "reset_done": False
            })
            
        while queue:
            # 2a. Check Minimum Chunks Floor BEFORE processing
            # If current chunk is too small (< 35 chars AND < 6 words), merge with next.
            # This handles both initial small chunks AND small chunks resulting from splits.
            
            # Peek current
            cur_item = queue[0]
            c_seg = cur_item["seg"]
            c_txt = c_seg.text.strip()
            
            # Criteria: < 35 chars AND < 6 words
            if len(c_txt) < 35 and len(c_txt.split()) < 6:
                # Attempt Merge
                if len(queue) > 1:
                    next_item = queue[1]
                    n_seg = next_item["seg"]
                    
                    # Merge Logic: Current + Next
                    merged_txt = (c_txt + " " + n_seg.text).strip()
                    # Inherit pause from next (as it's the end of the new block)
                    merged_pause = n_seg.pause_after
                    
                    print(f"    [MERGE] Chunk {cur_item['id']} too small ({len(c_txt)} chars). Merging with {next_item['id']}...")
                    
                    # Update Next item
                    # We modify queue[1] to contain the merged text
                    # We might want to keep the ID of the *first* matching chunk for consistency, 
                    # OR combine IDs. Let's combine IDs for clarity.
                    new_id = f"{cur_item['id']}+{next_item['id']}"
                    
                    # Create new segment
                    new_seg = ProcessingSegment(text=merged_txt, pause_after=merged_pause)
                    
                    queue[1]["seg"] = new_seg
                    # queue[1]["id"] = new_id # Optional: update ID
                    
                    # Pop current (discard it, as it's now in next)
                    queue.pop(0)
                    continue # Restart loop with the new merged item as head
                else:
                    # Last item is small. We must process it as is.
                    pass

            # 2b. Process Head
            item = queue.pop(0)
            seg = item["seg"]
            cid = item["id"]
            attempt = item["attempt"]
            forced_params = item["params"]
            reset_done = item.get("reset_done", False)  # Track if this item came from a reset
            
            # Determine params
            cur_temp = forced_params["temperature"] if forced_params else args.temperature
            cur_rep = forced_params["repetition_penalty"] if forced_params else args.repetition_penalty
            cur_top_p = forced_params["top_p"] if forced_params else args.top_p
            cur_top_k = forced_params["top_k"] if forced_params else args.top_k
            
            # Label for log
            label = "Gate"
            if attempt == 1: label = "Regen#1"
            elif attempt == 2: label = "Regen#2_SPLIT" if ("a" in str(cid) or "b" in str(cid)) else "Regen#2"

            if attempt == 1 and ("a" in str(cid) or "b" in str(cid)): label = "Regen#2_SPLIT" # Override if ID implies split
            
            if reset_done: label += "(RESET)"
            
            # Temp file
            temp_name = f"temp_{ep}_{lang}_{cid}_a{attempt}.wav"
            raw_seg_path = out_dir / temp_name
            
            if args.debug_metrics:
                 print(f"    [{label}] Processing {cid} (len={len(seg.text)})")

            tts_piece_to_wav(
                tts=tts,
                text=seg.text,
                speaker_wav=str(voice_path),
                language=lang,
                wav_path=raw_seg_path,
                sample_rate=args.wav_sr,
                temperature=cur_temp,
                repetition_penalty=cur_rep,
                top_p=cur_top_p,
                top_k=cur_top_k,
                trim_silence=args.trim_silence,
                debug_silence=args.debug_silence
            )
            
            # Metrics & Gate
            is_valid = False
            m = None
            reason = "File not created"
            action = "RETRY"
            
            if raw_seg_path.exists():
                m = analyze_audio_chunk(raw_seg_path, seg.text, args.wav_sr)
                if m:
                    is_valid, reason, action = check_audio_quality(m, seg.text)
                    
                    status_str = "OK" if is_valid else "BAD"
                    
                    log_line = (
                        f"    [{label}] idx={cid} "
                        f"dur={m.duration_s:.2f}s "
                        f"rms={m.rms_dbfs:.1f}dB "
                        f"sil%={m.internal_silence_pct:.1f}% "
                        f"wps={m.words_per_sec:.1f} "
                        f"-> {status_str}"
                    )
                    if not is_valid:
                        log_line += f" reason={reason}"
                    
                    print(log_line)
            
            if is_valid:
                # Success
                final_name = f"{ep}_{lang}_{cid}.wav"
                final_path = out_dir / final_name
                
                shutil.move(str(raw_seg_path), str(final_path))
                
                if args.normalize_wavs:
                    norm_path = out_dir / f"{ep}_{lang}_{cid}_norm.wav"
                    ffmpeg_normalize_wav(final_path, norm_path, args.wav_sr)
                    current_wav = norm_path
                else:
                    current_wav = final_path
                
                if m:
                    m.filename = final_path.name
                    seg_metrics.append(m)
                
                seg_wavs_final_list.append(current_wav)
                
                # Pause
                if seg.pause_after > 0.01:
                    sil_path = out_dir / f"silence_{float(seg.pause_after):.3f}s.wav"
                    if not sil_path.exists():
                        make_silence_wav(seg.pause_after, args.wav_sr, sil_path)
                    seg_wavs_final_list.append(sil_path)
                    
            else:
                # Failed
                # HANG CHECK
                if action == "RESET":
                    print(f"      -> [HANG] inference hang detected!")
                    
                    if reset_done or attempt >= 3: 
                        # Already reset or max attempts, give up -> DROP
                        print(f"      -> [DROP] Still hanging/failing after reset. Dropping chunk.")
                        failed_chunks_log.append(f"{cid} | {seg.text[:30]}... | HANG_DROP")
                        
                        # Generate 0.2s silence as replacement
                        drop_path = out_dir / f"{ep}_{lang}_{cid}_DROP.wav"
                        make_silence_wav(0.2, args.wav_sr, drop_path)
                        
                        # Add to list
                        if m: seg_metrics.append(m) # record bad metric
                        seg_wavs_final_list.append(drop_path)
                        
                        if seg.pause_after > 0.01:
                            sil_path = out_dir / f"silence_{float(seg.pause_after):.3f}s.wav"
                            if not sil_path.exists(): make_silence_wav(seg.pause_after, args.wav_sr, sil_path)
                            seg_wavs_final_list.append(sil_path)
                    else:
                        # Reset & Retry
                        # Perform Reset
                        tts = reset_xtts_model(tts, args.gpu)
                        
                        # Re-queue SAME segment with conservative params AND reset_done=True
                        print(f"      -> Retrying with fresh model (Reset)...")
                        queue.insert(0, {
                            "seg": seg, 
                            "id": cid, 
                            "attempt": attempt + 1,
                            "params": CONSERVATIVE_PARAMS,
                            "reset_done": True
                        })
                        
                else:
                    # Normal Retry
                    next_attempt = attempt + 1
                    
                    if next_attempt == 1:
                        # Retry 1: Conservative
                        print(f"      -> Retrying with conservative params...")
                        queue.insert(0, {
                            "seg": seg, "id": cid, "attempt": 1,
                            "params": CONSERVATIVE_PARAMS,
                            "reset_done": reset_done
                        })
                        
                    elif next_attempt == 2:
                        # Retry 2: Split
                        print(f"      -> Attempting split...")
                        t1, t2 = split_text_for_retry(seg.text)
                        
                        # Note: The "Start of Loop Check" will handle if t1 or t2 are too small
                        # But we should prefer NOT to split if we know they will be small.
                        # However, let's rely on the Merge logic at loop start to fix it if they are small.
                        # Except: if we split -> small -> merge -> original, we infinite loop?
                        # Yes.
                        # So we must check here too: if split produces chunks that would trigger mergeback to original, abort split.
                        
                        if t2 and len(t1) > 10 and len(t2) > 10:
                            s1 = ProcessingSegment(t1, args.sentence_silence)
                            s2 = ProcessingSegment(t2, seg.pause_after)
                            
                            id_a = f"{cid}a"
                            id_b = f"{cid}b"
                            
                            queue.insert(0, {"seg": s2, "id": id_b, "attempt": 1, "params": CONSERVATIVE_PARAMS, "reset_done": reset_done})
                            queue.insert(0, {"seg": s1, "id": id_a, "attempt": 1, "params": CONSERVATIVE_PARAMS, "reset_done": reset_done})
                            
                            print(f"      -> Split into {id_a} & {id_b}")
                        else:
                            print(f"      -> Split failed (too small). Accepting BAD chunk.")
                            failed_chunks_log.append(f"{cid} | {seg.text[:30]}... | {reason}")
                            
                            final_name = f"{ep}_{lang}_{cid}_BAD.wav"
                            final_path = out_dir / final_name
                            shutil.move(str(raw_seg_path), str(final_path))
                            
                            if args.normalize_wavs:
                                 norm_path = out_dir / f"{ep}_{lang}_{cid}_norm.wav"
                                 ffmpeg_normalize_wav(final_path, norm_path, args.wav_sr)
                                 current_wav = norm_path
                            else:
                                 current_wav = final_path
                            
                            if m:
                                 m.filename = final_path.name
                                 seg_metrics.append(m)
                            seg_wavs_final_list.append(current_wav)

                            if seg.pause_after > 0.01:
                                sil_path = out_dir / f"silence_{float(seg.pause_after):.3f}s.wav"
                                if not sil_path.exists(): make_silence_wav(seg.pause_after, args.wav_sr, sil_path)
                                seg_wavs_final_list.append(sil_path)

                    else:
                        # Retry limit
                        print(f"      -> Max retries. Accepting BAD chunk.")
                        failed_chunks_log.append(f"{cid} | {seg.text[:30]}... | {reason}")
                        final_name = f"{ep}_{lang}_{cid}_FAIL.wav"
                        final_path = out_dir / final_name
                        shutil.move(str(raw_seg_path), str(final_path))
                        
                        if args.normalize_wavs:
                             norm_path = out_dir / f"{ep}_{lang}_{cid}_norm.wav"
                             ffmpeg_normalize_wav(final_path, norm_path, args.wav_sr)
                             current_wav = norm_path
                        else:
                             current_wav = final_path
                        
                        if m:
                             m.filename = final_path.name
                             seg_metrics.append(m)
                        seg_wavs_final_list.append(current_wav)
                        
                        if seg.pause_after > 0.01:
                            sil_path = out_dir / f"silence_{float(seg.pause_after):.3f}s.wav"
                            if not sil_path.exists(): make_silence_wav(seg.pause_after, args.wav_sr, sil_path)
                            seg_wavs_final_list.append(sil_path)

        # 3. Concat Master
        
        # Save Metrics
        metrics_csv = out_dir / f"metrics_{ep}_{lang}.csv"
        save_metrics_csv(seg_metrics, metrics_csv)
        print_metrics_summary(seg_metrics)
        
        if failed_chunks_log:
            fail_txt = out_dir / f"fail_chunks_{ep}_{lang}.txt"
            fail_txt.write_text("\n".join(failed_chunks_log), encoding="utf-8")
            print(f"!!! WARNING: {len(failed_chunks_log)} broken chunks saved to {fail_txt}")
        
        if args.debug_metrics or args.debug_outliers:
            outlier_txt = out_dir / f"outliers_{ep}_{lang}.txt"
            analyze_outliers(seg_metrics, outlier_txt)

        master_wav = out_dir / f"{ep}_{lang}.master.wav"
        ffmpeg_concat_wavs_to_master_wav(seg_wavs_final_list, master_wav, args.wav_sr)
        
        # 4. Final Encode (Speed + MP3)
        out_mp3 = out_dir / f"{ep}_{lang}.mp3"
        ffmpeg_encode_mp3_from_wav(
            master_wav, 
            out_mp3, 
            args.mp3_bitrate, 
            args.mp3_vbr_q, 
            args.speed
        )
        
        print(f"OK -> {out_mp3}")
            
if __name__ == "__main__":
    main()
    ap.add_argument("--max_chars", type=int, default=250, help="Max chars per chunk")
    ap.add_argument("--min_chars", type=int, default=65, help="Min chars to merge")
    
    # Silence / Pacing
    ap.add_argument("--trim_silence", action="store_true", help="Trim start/end silence of chunks")
    ap.add_argument("--paragraph_silence", type=float, default=0.3, help="Silence after paragraph (sec)")
    ap.add_argument("--sentence_silence", type=float, default=0.1, help="Silence after chunk/sentence (sec)")

    # Output Format
    ap.add_argument("--wav_sr", type=int, default=24000)
    ap.add_argument("--normalize_wavs", action="store_true", help="Normalize WAV chunks")
    ap.add_argument("--mp3_bitrate", default="192k")
    ap.add_argument("--mp3_vbr_q", type=int, default=2)
    ap.add_argument("--speed", type=float, default=1.0)
    
    # Debug
    ap.add_argument("--debug_metrics", action="store_true")
    ap.add_argument("--debug_outliers", action="store_true", help="Analyze and print top outlier chunks")
    ap.add_argument("--debug_silence", action="store_true")

    args = ap.parse_args()

    ensure_deps()
    ensure_ffmpeg()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    narr_dir = Path(args.narr_dir).expanduser().resolve()
    voices_dir = Path(args.voices_dir).expanduser().resolve()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    if not langs: raise RuntimeError("Nenhum idioma")

    ep = args.episode.strip()
    
    # Load Model
    print(f"-> Loading model {MODEL_NAME}...")
    tts = TTS(MODEL_NAME)
    tts = tts.to("cuda" if args.gpu else "cpu")
    print("-> Model loaded.")
    
    # Conservative Params for Retry
    CONSERVATIVE_PARAMS = {
        "temperature": 0.50,
        "repetition_penalty": 1.05,
        "top_p": 0.75,
        "top_k": 30
    }

    for lang in langs:
        txt_path = narr_dir / f"{ep}_{lang}.txt"
        if not txt_path.exists():
            print(f"Skipping {lang}, file not found: {txt_path}")
            continue
            
        voice_path = get_voice_wav_for_lang(voices_dir, lang, args.wav_sr)
        
        print(f"\n[{lang.upper()}] Processing {txt_path.name}...")

        # 1. Read & Segment
        base_segments = read_smart_paragraphs(txt_path, args.paragraph_silence, args.sentence_silence)
        
        final_segments = process_segments(
            base_segments, 
            min_chars=args.min_chars, 
            max_chars=args.max_chars, 
            sentence_silence=args.sentence_silence
        )
        
        print(f"    -> {len(final_segments)} chunks ready.")
        


    # 2. Generate with Quality Gate Queue
        seg_wavs_final_list: List[Path] = []
        seg_metrics: List[AudioMetrics] = []
        failed_chunks_log: List[str] = []
        
        # Queue items: dict(seg, id, attempt, params, reset_done)
        queue = []
        # Populate queue
        for i, seg in enumerate(final_segments, start=1):
            queue.append({
                "seg": seg,
                "id": f"{i:03d}",
                "attempt": 0,
                "params": None,
                "reset_done": False
            })
            
        while queue:
            # PEEK first to check merge logic before pop?
            # User requirement: "never generate chunks smaller than min_retry_chars=35 ... merge with adjacent"
            # This applies to ANY chunk in queue before generation.
            
            # Use current item
            item = queue[0] 
            seg = item["seg"]
            cid = item["id"]
            
            # Check size
            # min_retry_chars=35 or min_retry_words=6
            t_len = len(seg.text)
            n_words = len(seg.text.split())
            
            if t_len < 35 and n_words < 6:
                # Too small! Merge with next if available.
                if len(queue) > 1:
                    # Merge with next
                    next_item = queue[1]
                    # We merge CURRENT into NEXT ? Or NEXT into CURRENT?
                    # "prefer merging forward". Current is i, Next is i+1.
                    # Merging Current+Next.
                    
                    merged_text = (seg.text + " " + next_item["seg"].text).strip()
                    # New pause is the pause of the *next* segment ideally? Or current?
                    # Usually pause_after of the *last* part.
                    new_pause = next_item["seg"].pause_after 
                    
                    # Update next item
                    # We discard current item and update next item to include current text
                    # Preserving ID? Maybe use ID of current?
                    # Let's say we extend next item.
                    # Or we extend current and discard next.
                    
                    # Log merge
                    print(f"    [MERGE] Chunk {cid} too small ({t_len} chars). Merging with {next_item['id']}...")
                    
                    # Update queue[1]
                    queue[1]["seg"].text = merged_text
                    # queue[1]["id"] = f"{cid}+{next_item['id']}" # optional complex ID?
                    # Keep ID of next or current? User didn't specify.
                    # Keeping next ID might skip metrics for current ID.
                    # Let's keep ID of current for tracking?
                    # Actually, if we merge, we effectively skip one generation.
                    # Best to pop current, update next.
                    
                    queue.pop(0)
                    continue # Loop again with next item as head
                    
                else:
                    # Last item and small?
                    # Just process it, nothing to merge forward with.
                    pass

            # Now pop for real processing
            item = queue.pop(0)
            seg = item["seg"]
            cid = item["id"]
            attempt = item["attempt"]
            forced_params = item["params"]
            reset_done = item.get("reset_done", False)

            # Determine params
            cur_temp = forced_params["temperature"] if forced_params else args.temperature
            cur_rep = forced_params["repetition_penalty"] if forced_params else args.repetition_penalty
            cur_top_p = forced_params["top_p"] if forced_params else args.top_p
            cur_top_k = forced_params["top_k"] if forced_params else args.top_k
            
            # Label
            label = "Gate"
            if attempt == 1: label = "Regen#1"
            elif attempt == 2: label = "Regen#2_SPLIT" if "a" in str(cid) or "b" in str(cid) else "Regen#2"
            if attempt == 1 and ("a" in cid or "b" in cid): label = "Regen#2_SPLIT"
            if reset_done: label += "(RESET)"
            
            temp_name = f"temp_{ep}_{lang}_{cid}_a{attempt}.wav"
            raw_seg_path = out_dir / temp_name
            
            if args.debug_metrics:
                 print(f"    [{label}] Processing {cid} (len={len(seg.text)})")

            tts_piece_to_wav(
                tts=tts,
                text=seg.text,
                speaker_wav=str(voice_path),
                language=lang,
                wav_path=raw_seg_path,
                sample_rate=args.wav_sr,
                temperature=cur_temp,
                repetition_penalty=cur_rep,
                top_p=cur_top_p,
                top_k=cur_top_k,
                trim_silence=args.trim_silence,
                debug_silence=args.debug_silence
            )
            
            is_valid = False
            m = None
            reason = "File not created"
            action = "RETRY"
            
            if raw_seg_path.exists():
                m = analyze_audio_chunk(raw_seg_path, seg.text, args.wav_sr)
                if m:
                    is_valid, reason, action = check_audio_quality(m, seg.text)
                    
                    status_str = "OK" if is_valid else "BAD"
                    
                    log_line = (
                        f"    [{label}] idx={cid} "
                        f"dur={m.duration_s:.2f}s "
                        f"rms={m.rms_dbfs:.1f}dB "
                        f"sil%={m.internal_silence_pct:.1f}% "
                        f"wps={m.words_per_sec:.1f} "
                        f"-> {status_str}"
                    )
                    if not is_valid:
                        log_line += f" reason={reason}"
                    
                    print(log_line)
            
            if is_valid:
                # Success Logic (Same as before)
                final_name = f"{ep}_{lang}_{cid}.wav"
                final_path = out_dir / final_name
                shutil.move(str(raw_seg_path), str(final_path))
                
                if args.normalize_wavs:
                    norm_path = out_dir / f"{ep}_{lang}_{cid}_norm.wav"
                    ffmpeg_normalize_wav(final_path, norm_path, args.wav_sr)
                    current_wav = norm_path
                else:
                    current_wav = final_path
                
                if m:
                    m.filename = final_path.name
                    seg_metrics.append(m)
                
                seg_wavs_final_list.append(current_wav)
                
                if seg.pause_after > 0.01:
                    sil_path = out_dir / f"silence_{float(seg.pause_after):.3f}s.wav"
                    if not sil_path.exists():
                        make_silence_wav(seg.pause_after, args.wav_sr, sil_path)
                    seg_wavs_final_list.append(sil_path)
                    
            else:
                # Failed
                # Check HANG ACTION
                if action == "RESET":
                    # HANG DETECTED
                    print(f"      -> [HANG] inference hang detected!")
                    
                    if reset_done:
                        # Already reset once, Drop it.
                        print(f"      -> [DROP] Still hanging after reset. Dropping chunk.")
                        failed_chunks_log.append(f"{cid} | {seg.text[:30]}... | HANG_SUSPECT_DROP")
                        
                        # Generate SILENCE replacement (0.2s)
                        sil_drop_path = out_dir / f"{ep}_{lang}_{cid}_DROP.wav"
                        make_silence_wav(0.2, args.wav_sr, sil_drop_path)
                        
                        # Add to final list as silence (placeholder)
                        seg_metrics.append(m) # keep bad metric?
                        seg_wavs_final_list.append(sil_drop_path)
                        
                        # Pause
                        if seg.pause_after > 0.01:
                            sil_path = out_dir / f"silence_{float(seg.pause_after):.3f}s.wav"
                            if not sil_path.exists(): make_silence_wav(seg.pause_after, args.wav_sr, sil_path)
                            seg_wavs_final_list.append(sil_path)
                            
                    else:
                        # Perform Reset
                        tts = reset_xtts_model(tts, args)
                        
                        # Re-queue SAME segment (attempt+1 for log, enforce conservative)
                        # Mark reset_done=True
                        print(f"      -> Retrying with fresh model...")
                        queue.insert(0, {
                            "seg": seg, "id": cid, 
                            "attempt": attempt + 1,
                            "params": CONSERVATIVE_PARAMS,
                            "reset_done": True
                        })

                else:
                    # Normal Retry Logic (Non-Hang)
                    next_attempt = attempt + 1
                    
                    if next_attempt == 1:
                        print(f"      -> Retrying with conservative params...")
                        queue.insert(0, {
                            "seg": seg, "id": cid, "attempt": 1,
                            "params": CONSERVATIVE_PARAMS,
                            "reset_done": reset_done
                        })
                        
                    elif next_attempt == 2:
                        print(f"      -> Attempting split...")
                        t1, t2 = split_text_for_retry(seg.text)
                        
                        # Check split validity vs Minimum Retry Floor (User said: "If split would go below...")
                        # If t1 or t2 < 35 chars, don't split?
                        # Actually loop top check handles "never generate chunks smaller".
                        # But if we insert them here, the loop top will catch them next iteration!
                        # However, user says "If split would go below... merge with adjacent".
                        # But here we are *creating* new chunks.
                        # If we split and they are small, the loop top check will see them and try to merge them.
                        # E.g. T1 (small) -> Merges with T2?
                        # Yes, queue[0] will be T1. queue[1] will be T2.
                        # T1 < 35 -> Merges with T2. Result = Original Text.
                        # Infinite loop? 
                        # If Split -> Small -> Merge -> Original -> Fail -> Split -> Small...
                        # We need to detect if split is futile.
                        
                        if t2 and (len(t1) >= 35 or len(t1.split()) >= 6) and (len(t2) >= 35 or len(t2.split()) >= 6):
                            s1 = ProcessingSegment(t1, args.sentence_silence)
                            s2 = ProcessingSegment(t2, seg.pause_after)
                            
                            id_a = f"{cid}a"
                            id_b = f"{cid}b"
                            
                            queue.insert(0, {"seg": s2, "id": id_b, "attempt": 1, "params": CONSERVATIVE_PARAMS, "reset_done": reset_done})
                            queue.insert(0, {"seg": s1, "id": id_a, "attempt": 1, "params": CONSERVATIVE_PARAMS, "reset_done": reset_done})
                            
                            print(f"      -> Split into {id_a} & {id_b}")
                        else:
                            # Split yields tiny chunks, abort split.
                            print(f"      -> Split cancelled (chunks too small). Accepting BAD chunk.")
                            failed_chunks_log.append(f"{cid} | {seg.text[:30]}... | {reason}")
                            
                            final_name = f"{ep}_{lang}_{cid}_BAD.wav"
                            final_path = out_dir / final_name
                            shutil.move(str(raw_seg_path), str(final_path))
                            
                            if args.normalize_wavs:
                                 norm_path = out_dir / f"{ep}_{lang}_{cid}_norm.wav"
                                 ffmpeg_normalize_wav(final_path, norm_path, args.wav_sr)
                                 current_wav = norm_path
                            else:
                                 current_wav = final_path
                            
                            if m:
                                 m.filename = final_path.name
                                 seg_metrics.append(m)
                            seg_wavs_final_list.append(current_wav)

                            if seg.pause_after > 0.01:
                                sil_path = out_dir / f"silence_{float(seg.pause_after):.3f}s.wav"
                                if not sil_path.exists(): make_silence_wav(seg.pause_after, args.wav_sr, sil_path)
                                seg_wavs_final_list.append(sil_path)

                    else:
                        print(f"      -> Max retries. Accepting BAD chunk.")
                        failed_chunks_log.append(f"{cid} | {seg.text[:30]}... | {reason}")
                        final_name = f"{ep}_{lang}_{cid}_FAIL.wav"
                        final_path = out_dir / final_name
                        shutil.move(str(raw_seg_path), str(final_path))
                        
                        if args.normalize_wavs:
                             norm_path = out_dir / f"{ep}_{lang}_{cid}_norm.wav"
                             ffmpeg_normalize_wav(final_path, norm_path, args.wav_sr)
                             current_wav = norm_path
                        else:
                             current_wav = final_path
                        
                        if m:
                             m.filename = final_path.name
                             seg_metrics.append(m)
                        seg_wavs_final_list.append(current_wav)
                        
                        if seg.pause_after > 0.01:
                            sil_path = out_dir / f"silence_{float(seg.pause_after):.3f}s.wav"
                            if not sil_path.exists(): make_silence_wav(seg.pause_after, args.wav_sr, sil_path)
                            seg_wavs_final_list.append(sil_path)

            
if __name__ == "__main__":
    main()
