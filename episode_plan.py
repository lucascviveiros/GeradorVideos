#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".m4v", ".webm"}

# Fallback em duas camadas (segunda camada fixa)
SECONDARY_FALLBACK_TAG = "architecture"

# Máximo de tags/cortes que você quer consumir por bloco [TAGS B-ROLL]
EDITOR_MAX_CUTS_PER_BLOCK = 4

# --------------------------
# Similar tags (sinônimos)
# --------------------------
SIMILAR_TAGS: Dict[str, List[str]] = {
    "abstract": ["abstract_dark", "reflection", "perspective", "matrix"],
    "abstract_dark": ["abstract", "worry", "stress", "repetition_loop"],
    "adventure": ["travel", "opportunity", "growth", "nature"],
    "alone": ["isolation", "sad", "worry", "reflection"],
    "appreciation": ["thanks", "kindness", "family", "peace"],
    "architecture": ["construction", "city", "business", "abstract_dark"],
    "bad_posture": ["working", "stress", "fatigue", "illness"],
    "brain": ["decision", "reflection", "discussion", "perspective"],
    "broken": ["stress", "sad", "risk", "death"],
    "business": ["working", "office_night", "city", "tecnology"],
    "calm": ["relax", "peace", "nature", "self_care"],
    "christmas": ["family", "appreciation", "thanks", "happy"],
    "city": ["urban_lifestyle", "working", "office_night", "subway"],
    "clean": ["self_care", "calm", "morning", "peace"],
    "cold": ["alone", "abstract_dark", "sad", "nature"],
    "construction": ["architecture", "city", "hands_on", "business"],
    "cooperation": ["business", "discussion", "kindness", "hands_on"],
    "creation": ["growth", "hands_on", "music", "perspective"],
    "data": ["tecnology", "business", "multitask", "working"],
    "death": ["illness", "broken", "sad", "reflection"],
    "debt_pressure": ["financial_pressure", "expenses", "worry", "credit_card"],
    "decision": ["brain", "risk", "perspective", "contract_signing"],
    "denial": ["decision", "worry", "stress", "abstract_dark"],
    "dice": ["risk", "decision", "opportunity", "money_disappearing"],
    "discussion": ["business", "decision", "brain", "cooperation"],
    "doubt": ["decision", "worry", "reflection", "perspective"],
    "energy": ["morning", "working", "training", "growth"],
    "expenses": ["spent", "money_disappearing", "financial_pressure", "debt_pressure"],
    "family": ["appreciation", "thanks", "christmas", "happy"],
    "fatigue": ["stress", "working", "worry", "bad_posture"],
    "fire": ["risk", "broken", "abstract_dark", "news"],
    "food": ["morning", "self_care", "calm", "walk"],
    "freedom": ["peace", "relax", "nature", "happy"],
    "growth": ["opportunity", "training", "prosperity", "creation"],
    "hands_on": ["creation", "construction", "training", "business"],
    "happy": ["peace", "freedom", "appreciation", "prosperity"],
    "illness": ["stress", "fatigue", "self_care", "death"],
    "investments": ["prosperity", "risk", "financial_pressure", "decision"],
    "isolation": ["alone", "sad", "reflection", "nature"],
    "kindness": ["appreciation", "thanks", "self_care", "peace"],
    "manifestation": ["prosperity", "growth", "spiritual", "matrix"],
    "matrix": ["abstract", "abstract_dark", "tecnology", "data"],
    "morning": ["calm", "preparation", "self_care", "walk"],
    "movie": ["reflection", "perspective", "music", "abstract"],
    "multitask": ["working", "office_night", "data", "stress"],
    "music": ["calm", "reflection", "movie", "nature"],
    "nature": ["peace", "calm", "walk", "freedom"],
    "news": ["business", "city", "tecnology", "risk"],
    "noise": ["city", "office_night", "stress", "repetition_loop"],
    "office_night": ["working", "city", "multitask", "stress"],
    "opportunity": ["growth", "business", "decision", "prosperity"],
    "peace": ["calm", "relax", "nature", "freedom"],
    "perspective": ["reflection", "decision", "brain", "peace"],
    "phone": ["tecnology", "credit_card", "money_disappearing", "urban_lifestyle"],
    "preparation": ["morning", "studying", "training", "decision"],
    "prosperity": ["investments", "growth", "freedom", "happy"],
    "reflection": ["brain", "perspective", "alone", "calm"],
    "relax": ["self_care", "calm", "peace", "nature"],
    "repetition_loop": ["waiting_in_line", "train", "subway", "noise"],
    "risk": ["dice", "decision", "future_burden", "investments"],
    "robber": ["risk", "city", "abstract_dark", "money_disappearing"],
    "sad": ["alone", "isolation", "worry", "fatigue"],
    "self_care": ["relax", "calm", "kindness", "walk"],
    "spent": ["expenses", "money_disappearing", "credit_card", "financial_pressure"],
    "spiritual": ["reflection", "peace", "manifestation", "nature"],
    "stress": ["worry", "fatigue", "working", "office_night"],
    "studying": ["preparation", "brain", "working", "reflection"],
    "subway": ["train", "city", "waiting_in_line", "urban_lifestyle"],
    "sync": ["repetition_loop", "multitask", "music", "data"],
    "tecnology": ["data", "business", "phone", "news"],
    "thanks": ["appreciation", "kindness", "family", "peace"],
    "thirst": ["food", "self_care", "walk", "morning"],
    "time": ["reflection", "working", "preparation", "repetition_loop"],
    "train": ["subway", "waiting_in_line", "city", "repetition_loop"],
    "training": ["growth", "energy", "hands_on", "preparation"],
    "travel": ["adventure", "freedom", "nature", "opportunity"],
    "waiting_in_line": ["repetition_loop", "subway", "train", "stress"],
    "walk": ["nature", "self_care", "morning", "peace"],
    "weight": ["training", "self_care", "stress", "food"],
    "working": ["office_night", "business", "multitask", "stress"],
    "worry": ["stress", "debt_pressure", "financial_pressure", "alone"],
    "credit_card": ["expenses", "money_disappearing", "contract_signing", "debt_pressure"],
    "future_burden": ["debt_pressure", "risk", "financial_pressure", "reflection"],
    "contract_signing": ["credit_card", "decision", "future_burden", "risk"],
    "money_disappearing": ["spent", "expenses", "credit_card", "financial_pressure"],
    "urban_lifestyle": ["city", "working", "subway", "money_disappearing"],
    "financial_pressure": ["debt_pressure", "worry", "expenses", "money_disappearing"],
}

# --------------------------
# TAG timing profiles (Editor mode)
# --------------------------
TAG_DURATION_PROFILES: Dict[str, Tuple[float, float]] = {
    "office_night": (3.5, 5.5),
    "worry": (3.5, 5.5),
    "stress": (3.0, 5.0),
    "reflection": (3.5, 6.0),
    "alone": (3.5, 6.0),
    "isolation": (3.5, 6.0),
    "sad": (3.5, 6.0),

    "business": (2.5, 4.0),
    "contract_signing": (2.5, 4.0),
    "phone": (2.0, 3.5),
    "data": (2.0, 3.5),
    "money_disappearing": (2.5, 4.0),
    "expenses": (2.5, 4.0),
    "credit_card": (2.5, 4.0),
    "financial_pressure": (3.0, 5.0),
    "debt_pressure": (3.0, 5.0),

    "time": (2.5, 4.5),
    "working": (2.5, 4.5),
    "repetition_loop": (2.5, 4.5),
    "waiting_in_line": (2.5, 4.5),
    "city": (2.5, 4.5),
    "subway": (2.5, 4.5),
    "train": (2.5, 4.5),

    "decision": (2.5, 4.5),
    "risk": (2.5, 4.5),
    "future_burden": (3.0, 5.0),
    "denial": (3.0, 5.0),
    "perspective": (3.0, 5.5),

    "studying": (2.5, 4.5),
    "preparation": (2.5, 4.5),
    "growth": (2.5, 4.5),
    "investments": (2.5, 4.5),
    "prosperity": (2.5, 4.5),
    "tecnology": (2.0, 3.5),

    "fatigue": (3.0, 5.0),
    "bad_posture": (3.0, 5.0),
    "illness": (3.0, 5.0),
    "self_care": (3.0, 5.0),
}

DEFAULT_TAG_RANGE = (2.8, 4.6)
EDITOR_JITTER = 0.35

# --------------------------
# Data models
# --------------------------
@dataclass
class ScriptBlock:
    tags: Optional[List[str]]  # tags vindas do roteiro, sem '#'
    text: str


@dataclass
class Scene:
    tag: str
    text: str
    duration: float
    clip_path: Path
    forced_tags: Optional[List[str]] = None


# --------------------------
# Parse do roteiro: [TAGS B-ROLL]
# --------------------------
TAGS_LINE_RE = re.compile(
    r"^\s*\[?\s*TAGS(?:\s+B-ROLL)?\s*\]?\s*:\s*(.*?)\s*$", re.IGNORECASE
)


def parse_script_blocks(text_path: Path) -> List[ScriptBlock]:
    """
    Se houver pelo menos uma linha TAGS, entra em modo tagueado.
    Texto antes da primeira TAG é ignorado.
    """
    if not text_path.is_file():
        raise RuntimeError(f"Roteiro não encontrado: {text_path}")

    raw = (
        text_path.read_text(encoding="utf-8", errors="ignore")
        .replace("\r\n", "\n")
        .strip("\n")
    )
    lines = raw.split("\n")

    blocks: List[ScriptBlock] = []
    cur_tags: Optional[List[str]] = None
    cur_text_lines: List[str] = []
    saw_any_tags = False

    def normalize_tag(t: str) -> str:
        t = (t or "").strip()
        if t.startswith("#"):
            t = t[1:]
        t = t.strip().replace(" ", "_")
        return t

    def flush() -> None:
        nonlocal cur_tags, cur_text_lines
        txt = "\n".join(cur_text_lines).strip()
        if txt:
            blocks.append(ScriptBlock(tags=cur_tags, text=txt))
        cur_tags = None
        cur_text_lines = []

    for ln in lines:
        m = TAGS_LINE_RE.match(ln)
        if m:
            tag_part = (m.group(1) or "").strip()
            if not saw_any_tags:
                saw_any_tags = True
                cur_text_lines = []
                blocks = []
            else:
                flush()

            parts = [p.strip() for p in tag_part.split(",") if p.strip()]
            tags = [normalize_tag(p) for p in parts if normalize_tag(p)]
            cur_tags = tags if tags else None
            continue

        if not saw_any_tags:
            # Ignora texto antes da primeira linha de TAG
            continue

        cur_text_lines.append(ln)

    if saw_any_tags:
        flush()
        return [
            ScriptBlock(tags=b.tags, text=re.sub(r"\n{3,}", "\n\n", b.text.strip()))
            for b in blocks
        ]

    # fallback: modo antigo (sem TAGS) por parágrafos
    paras = read_paragraphs(text_path)
    return [ScriptBlock(tags=None, text=p) for p in paras]


def read_paragraphs(text_path: Path) -> List[str]:
    raw = text_path.read_text(encoding="utf-8", errors="ignore").strip()
    parts = re.split(r"\n\s*\n+", raw)
    paras = [re.sub(r"\s+", " ", p.strip()) for p in parts if p.strip()]
    if not paras:
        raise RuntimeError("Nenhum parágrafo encontrado no roteiro.")
    return paras


# --------------------------
# Tags.yaml + scoring
# --------------------------
def _norm(s: str) -> str:
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


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


def word_count(s: str) -> int:
    return len(re.findall(r"\b[\wÀ-ÿ]+\b", s))


def score_tag(paragraph: str, keywords: List[str]) -> int:
    p = _norm(paragraph)
    return sum(1 for kw in keywords if kw and kw in p)


def choose_tag(
    paragraph: str,
    tag_order: List[str],
    tags: Dict[str, List[str]],
    fallback: str,
) -> str:
    best_tag = fallback
    best_score = -1
    for tag in tag_order:
        s = score_tag(paragraph, tags.get(tag, []))
        if s > best_score:
            best_score = s
            best_tag = tag
    return best_tag if best_score > 0 else fallback


# --------------------------
# Indexação de clips
# --------------------------
PROVIDER_PREFIXES = ("pixabay_", "pexels_", "mix_", "stock_")


def _strip_provider_prefix(folder_name: str) -> str:
    for p in PROVIDER_PREFIXES:
        if folder_name.startswith(p):
            return folder_name[len(p):]
    return folder_name


def index_clips(clips_root: Path) -> Dict[str, List[Path]]:
    if not clips_root.is_dir():
        raise RuntimeError(f"Pasta raiz de clipes não existe: {clips_root}")

    index: Dict[str, List[Path]] = {}

    def add(key: str, files: List[Path]) -> None:
        if not key:
            return
        index.setdefault(key, []).extend(files)

    for sub in clips_root.iterdir():
        if not sub.is_dir():
            continue

        files = [
            f
            for f in sub.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS
        ]
        files.sort(key=lambda x: x.name.lower())
        if not files:
            continue

        folder = sub.name
        add(folder, files)

        no_prefix = _strip_provider_prefix(folder)
        if no_prefix != folder:
            add(no_prefix, files)

            parts = no_prefix.split("_")
            if len(parts) >= 2:
                add("_".join(parts[:2]), files)

            add(parts[0], files)

    # dedup por caminho dentro de cada key
    for k in list(index.keys()):
        seen = set()
        uniq: List[Path] = []
        for p in index[k]:
            rp = str(p.resolve())
            if rp not in seen:
                seen.add(rp)
                uniq.append(p)
        uniq.sort(key=lambda x: x.name.lower())
        index[k] = uniq

    if not index:
        raise RuntimeError(
            f"Nenhum vídeo encontrado em subpastas de {clips_root}.\n"
            f"Esperado: {clips_root}/tag1/*.mp4, {clips_root}/tag2/*.mp4 ..."
        )
    return index


# --------------------------
# Helpers editoriais
# --------------------------
def _dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def select_forced_tags_for_block(forced_tags: List[str]) -> List[str]:
    ft = [t for t in (forced_tags or []) if (t or "").strip()]
    ft = _dedup_keep_order(ft)
    return ft[:EDITOR_MAX_CUTS_PER_BLOCK]


def split_text_into_n_chunks(text: str, n: int) -> List[str]:
    text = (text or "").strip()
    if n <= 1:
        return [re.sub(r"\s+", " ", text).strip()] if text else [""]

    sentences = re.split(
        r'(?<!\d)(?<=[.!?])\s+(?=(?:["“”‘’(\[]\s*)?[A-Za-zÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇÑ])',
        text,
    )
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) >= n:
        buckets: List[List[str]] = [[] for _ in range(n)]
        counts = [0] * n

        def wc(s: str) -> int:
            return len(re.findall(r"\b[\wÀ-ÿ]+\b", s))

        for s in sentences:
            i = min(range(n), key=lambda k: counts[k])
            buckets[i].append(s)
            counts[i] += wc(s)

        out = [" ".join(b).strip() for b in buckets]
        return [re.sub(r"\s+", " ", o).strip() for o in out]

    # fallback por palavras
    words = re.findall(r"\S+", text)
    if not words:
        return [""] * n

    out: List[str] = []
    base = len(words) // n
    rem = len(words) % n
    start = 0
    for i in range(n):
        take = base + (1 if i < rem else 0)
        out.append(" ".join(words[start : start + take]).strip())
        start += take

    return [re.sub(r"\s+", " ", o).strip() for o in out]


def pick_scene_duration_for_tag(
    tag: str,
    rng: random.Random,
    *,
    min_scene: float,
    max_scene: float,
) -> float:
    lo, hi = TAG_DURATION_PROFILES.get(tag, DEFAULT_TAG_RANGE)
    d = rng.uniform(lo, hi) + rng.uniform(-EDITOR_JITTER, EDITOR_JITTER)
    d = min(max(d, min_scene), max_scene)
    return float(d)


def fit_durations_with_bounds(
    durs: List[float],
    total: float,
    minv: float,
    maxv: float,
    eps: float = 1e-6,
    max_iter: int = 60,
) -> List[float]:
    """
    Ajusta durações para somarem 'total' respeitando [minv, maxv],
    sem achatar tudo via rescale global simples.
    """
    n = len(durs)
    if n == 0:
        return []

    out = [float(x) for x in durs]
    free = set(range(n))

    for _ in range(max_iter):
        fixed_sum = sum(out[i] for i in range(n) if i not in free)
        remain = total - fixed_sum

        if remain <= eps:
            out = [minv for _ in range(n)]
            drift = total - sum(out)
            if out:
                out[-1] = max(0.01, out[-1] + drift)
            return out

        free_sum = sum(out[i] for i in free)
        if free_sum <= eps:
            base = remain / max(1, len(free))
            for i in free:
                out[i] = base
            free_sum = sum(out[i] for i in free) or eps

        scale = remain / free_sum

        changed = False
        for i in list(free):
            out[i] *= scale
            if out[i] < minv:
                out[i] = minv
                free.remove(i)
                changed = True
            elif out[i] > maxv:
                out[i] = maxv
                free.remove(i)
                changed = True

        if not changed:
            break

    drift = total - sum(out)
    if out and abs(drift) > 1e-4:
        out[-1] = max(0.01, out[-1] + drift)
    return out


def build_scenes(
    blocks: List[ScriptBlock],
    audio_dur: float,
    tag_order: List[str],
    tags: Dict[str, List[str]],
    clips_root: Path,
    fallback_tag: str,
    min_scene: float,
    max_scene: float,
    seed: int,
    tag_mode: str = "editor",   # editor|text (text = usa word_count / editor = profiles)
) -> List[Scene]:
    rng = random.Random(seed)
    clips_index = index_clips(clips_root)

    # controla arquivos já usados (path real) E nomes já usados (para evitar duplicar o mesmo vídeo)
    used_paths: set[str] = set()
    used_names: set[str] = set()

    def pick_unique_clip(desired_tag: str) -> Tuple[str, Path]:
        """
        Resolve desired_tag -> (resolved_tag, clip_path) com:
          - sinônimos + fallback em cascata
          - anti-repeat global por arquivo e por nome
        """
        candidates = [desired_tag]
        candidates += SIMILAR_TAGS.get(desired_tag, [])
        candidates += [SECONDARY_FALLBACK_TAG, fallback_tag]

        for t in _dedup_keep_order([c for c in candidates if (c or "").strip()]):
            pool = clips_index.get(t) or []
            if not pool:
                continue

            avail: List[Path] = []
            for p in pool:
                rp = str(p.resolve())
                name_id = p.name.lower()
                if rp in used_paths or name_id in used_names:
                    continue
                avail.append(p)

            if not avail:
                continue

            chosen = rng.choice(avail)
            rp_chosen = str(chosen.resolve())
            name_id_chosen = chosen.name.lower()
            used_paths.add(rp_chosen)
            used_names.add(name_id_chosen)
            return (t, chosen)

        raise RuntimeError(
            f"Sem clipes únicos disponíveis para '{desired_tag}' (nem sinônimos/fallbacks). "
            f"Você precisa: (a) mais inventário, ou (b) reduzir cortes por bloco."
        )

    # 1) Duração base por BLOCO (proporcional a palavras)
    texts = [b.text for b in blocks]
    counts = [max(1, word_count(t)) for t in texts]
    total_words = sum(counts) or 1

    raw_block_durs = [(c / total_words) * audio_dur for c in counts]
    # jitter leve pra não ficar robótico
    raw_block_durs = [d * rng.uniform(0.90, 1.10) for d in raw_block_durs]

    n_blocks = max(1, len(texts))
    avg = audio_dur / n_blocks if audio_dur > 0 else min_scene

    # min_scene efetivo pra não “estourar” quando há muitos blocos
    min_scene_eff = max(0.3, min(min_scene, avg * 0.90))
    max_scene_eff = max(max_scene, min_scene_eff + 0.2)

    block_durs = fit_durations_with_bounds(
        raw_block_durs,
        audio_dur,
        min_scene_eff,
        min(max_scene_eff, avg * 1.80),
    )

    # 1.1) Budget de cenas: quantas cenas cabem no áudio respeitando min_scene_eff
    min_for_budget = max(0.3, min_scene_eff)
    if audio_dur > 0 and min_for_budget > 0:
        raw_max_scenes = int(audio_dur // min_for_budget)
    else:
        raw_max_scenes = len(blocks) or 1

    # Budget global de cenas
    max_scenes = max(1, raw_max_scenes)

    # 2) Expande blocos em cenas (usando tags do bloco enquanto couber no budget)
    scenes: List[Scene] = []
    total_blocks = len(blocks)

    for idx, (b, block_dur) in enumerate(zip(blocks, block_durs)):
        remaining_blocks = total_blocks - idx
        remaining_budget = max_scenes - len(scenes)

        if remaining_budget <= 0:
            # Sem budget de cena: áudio continua, mas não criamos novos cortes
            break

        if b.tags:
            forced_all = select_forced_tags_for_block(b.tags)
            if not forced_all:
                # Cai para o modo keywords se a lista de tags vier vazia
                desired = choose_tag(b.text, tag_order, tags, fallback=fallback_tag)
                resolved_tag, clip_path = pick_unique_clip(desired)
                scenes.append(
                    Scene(
                        tag=resolved_tag,
                        text=b.text,
                        duration=float(block_dur),
                        clip_path=clip_path,
                        forced_tags=None,
                    )
                )
                continue

            # Máximo de cenas que esse bloco pode consumir sem matar
            # a possibilidade de 1 cena mínima para os blocos restantes
            max_for_this_block = remaining_budget - (remaining_blocks - 1)
            max_for_this_block = max(1, max_for_this_block)
            max_for_this_block = min(
                max_for_this_block,
                len(forced_all),
                EDITOR_MAX_CUTS_PER_BLOCK,
            )

            forced = forced_all[:max_for_this_block]
            n = max(1, len(forced))
            chunks = split_text_into_n_chunks(b.text, n)

            if tag_mode.lower() == "editor":
                durs = [
                    pick_scene_duration_for_tag(
                        t,
                        rng,
                        min_scene=min_scene_eff,
                        max_scene=max_scene_eff,
                    )
                    for t in forced
                ]
            else:
                # text-mode: divide proporcionalmente por palavras
                wcs = [max(1, word_count(x)) for x in chunks]
                s = sum(wcs) or 1
                durs = [(w / s) * float(block_dur) for w in wcs]

            for t, tx, dur in zip(forced, chunks, durs):
                resolved_tag, clip_path = pick_unique_clip(t)
                scenes.append(
                    Scene(
                        tag=resolved_tag,
                        text=tx,
                        duration=float(dur),
                        clip_path=clip_path,
                        forced_tags=b.tags,
                    )
                )
        else:
            # keywords mode: sempre 1 cena por bloco (desde que haja budget)
            desired = choose_tag(b.text, tag_order, tags, fallback=fallback_tag)
            resolved_tag, clip_path = pick_unique_clip(desired)
            scenes.append(
                Scene(
                    tag=resolved_tag,
                    text=b.text,
                    duration=float(block_dur),
                    clip_path=clip_path,
                    forced_tags=None,
                )
            )

    # 3) Ajusta durações finais para bater o áudio e respeitar bounds
    if scenes:
        final_durs = fit_durations_with_bounds(
            [s.duration for s in scenes],
            total=audio_dur,
            minv=min_scene_eff,
            maxv=max_scene_eff,
        )
        for sc, d in zip(scenes, final_durs):
            sc.duration = float(d)

        # drift final
        drift = audio_dur - sum(s.duration for s in scenes)
        if abs(drift) > 0.02:
            scenes[-1].duration = max(0.3, scenes[-1].duration + drift)

    return scenes
