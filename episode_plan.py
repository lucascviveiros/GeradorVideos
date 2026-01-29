#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import yaml

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".m4v", ".webm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

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
    block_id: Optional[int] = None      # 001, 002, ...
    core_idea: Optional[str] = None     # linha "Core idea: ..."

@dataclass
class Scene:
    tag: str
    text: str
    duration: float
    clip_path: Path
    forced_tags: Optional[List[str]] = None
    # NOVO: tag desejada originalmente (para log azul quando houver sinônimo)
    requested_tag: Optional[str] = None


# --------------------------
# Parse do roteiro: [TAGS B-ROLL]
# --------------------------
TAGS_LINE_RE = re.compile(
    r"^\s*\[?\s*TAGS(?:\s+B-ROLL)?\s*\]?\s*:\s*(.*?)\s*$", re.IGNORECASE
)


SCRIPT_BLOCK_HEADER_RE = re.compile(r"^\s*\[SCRIPT_BLOCKS\]\s*$", re.IGNORECASE)
BLOCK_ID_RE = re.compile(r"^\s*(\d{3})\s*$")
CORE_IDEA_RE = re.compile(r"^\s*Core idea:\s*(.*)$", re.IGNORECASE)


def parse_script_blocks(text_path: Path) -> List[ScriptBlock]:
    """
    - Se houver [SCRIPT_BLOCKS]: usa modo de blocos numerados (001, 002, ...).
    - Caso contrário:
        - Se houver TAGS: modo tagueado.
        - Senão: fallback por parágrafos.
    """
    if not text_path.is_file():
        raise RuntimeError(f"Roteiro não encontrado: {text_path}")

    raw = (
        text_path.read_text(encoding="utf-8", errors="ignore")
        .replace("\r\n", "\n")
        .strip("\n")
    )
    lines = raw.split("\n")

    # -----------------------------
    # 1) Modo [SCRIPT_BLOCKS]
    # -----------------------------
    has_script_blocks = any(SCRIPT_BLOCK_HEADER_RE.match(ln) for ln in lines)
    if has_script_blocks:
        blocks: List[ScriptBlock] = []
        cur_id: Optional[int] = None
        cur_core: Optional[str] = None
        cur_lines: List[str] = []
        in_section = False

        def flush_block():
            nonlocal cur_id, cur_core, cur_lines
            if cur_id is None:
                return
            text = "\n".join(cur_lines).strip()
            if not text:
                return
            blocks.append(
                ScriptBlock(
                    tags=None,
                    text=text,
                    block_id=cur_id,
                    core_idea=(cur_core.strip() if cur_core else None),
                )
            )
            cur_id = None
            cur_core = None
            cur_lines = []

        for ln in lines:
            if SCRIPT_BLOCK_HEADER_RE.match(ln):
                in_section = True
                continue
            if not in_section:
                continue

            m_id = BLOCK_ID_RE.match(ln)
            if m_id:
                flush_block()
                cur_id = int(m_id.group(1))
                cur_core = None
                cur_lines = []
                continue

            m_core = CORE_IDEA_RE.match(ln)
            if m_core:
                cur_core = m_core.group(1).strip()
                continue

            if ln.strip() == "":
                continue
            cur_lines.append(ln)

        flush_block()

        if not blocks:
            raise RuntimeError("[SCRIPT_BLOCKS] detectado, mas nenhum bloco foi parseado.")

        for b in blocks:
            b.text = re.sub(r"\n{3,}", "\n\n", b.text.strip())

        return blocks

    # -----------------------------
    # 2) Modo TAGS (comportamento antigo)
    # -----------------------------
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

    def flush_tag_block() -> None:
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
                flush_tag_block()

            parts = [p.strip() for p in tag_part.split(",") if p.strip()]
            tags = [normalize_tag(p) for p in parts if normalize_tag(p)]
            cur_tags = tags if tags else None
            continue

        if not saw_any_tags:
            continue

        cur_text_lines.append(ln)

    if saw_any_tags:
        flush_tag_block()
        return [
            ScriptBlock(tags=b.tags, text=re.sub(r"\n{3,}", "\n\n", b.text.strip()))
            for b in blocks
        ]

    # -----------------------------
    # 3) Fallback: parágrafos
    # -----------------------------
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
# Indexação de SEQUÊNCIA (modo sequência)
# --------------------------
def index_sequence_media(seq_dir: Path, mode: str) -> List[Path]:
    """
    mode: "image" ou "video"

    Espera arquivos com nome começando por número:
      - 1.png / 2.jpg / 003.webp
      - 1.mp4 / 02.mov / 10.mkv

    Ordena pelo número do prefixo (e por nome como desempate).

    NOVO COMPORTAMENTO:
      - Prioriza o número logo após o primeiro "_" no stem.
        Ex: "1_001__Anchor_A2__..." -> usa 001.
      - Fallbacks:
        (1) usa o número antes do primeiro "_" (comportamento antigo)
        (2) usa o primeiro número encontrado no início do stem (ex: "42algumacoisa")
    """
    if not seq_dir.is_dir():
        raise RuntimeError(f"Pasta de sequência não existe: {seq_dir}")

    if mode not in {"image", "video"}:
        raise RuntimeError(f"mode inválido: {mode} (use 'image' ou 'video')")

    exts = IMAGE_EXTS if mode == "image" else VIDEO_EXTS

    items: List[Tuple[int, Path]] = []
    for p in seq_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue

        parts = p.stem.split("_")

        # 1) NOVO: pega o número depois do primeiro "_" (parts[1])
        head2 = parts[1].strip() if len(parts) >= 2 else ""
        m2 = re.match(r"^\d+$", head2)
        if m2:
            items.append((int(head2), p))
            continue

        # 2) Fallback: comportamento antigo (antes do primeiro "_")
        head0 = parts[0].strip() if parts else p.stem.strip()
        m0 = re.match(r"^\d+$", head0)
        if m0:
            items.append((int(head0), p))
            continue

        # 3) Fallback final: ainda aceita "42algumacoisa" se existir
        m3 = re.match(r"^\s*(\d+)", p.stem)
        if not m3:
            continue
        items.append((int(m3.group(1)), p))

    items.sort(key=lambda t: (t[0], t[1].name.lower()))
    files = [p for _, p in items]

    if not files:
        raise RuntimeError(
            f"Nenhum arquivo de sequência encontrado em {seq_dir}.\n"
            f"Esperado: 1.png/2.jpg... (image) ou 1.mp4/2.mov... (video), sempre iniciando por número."
        )
    return files



# --------------------------
# Indexação de clips (tags -> vídeos)
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
        out.append(" ".join(words[start: start + take]).strip())
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


# --------------------------
# Builder original (tags -> vídeos)
# --------------------------
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
    used_paths_in: Optional[Set[str]] = None,  # <-- NOVO (histórico por idioma vindo do make_episode)
    history_policy: str = "strict",
    tag_mode: str = "text",   # editor|text (text = usa word_count / editor = profiles)
    avoid_tags: Optional[Set[str]] = None
) -> List[Scene]:
    rng = random.Random(seed)
    clips_index = index_clips(clips_root)

    # normaliza avoid_tags (case-insensitive)
    avoid_lc: Set[str] = set(x.lower() for x in (avoid_tags or set()) if x)

    # controla arquivos já usados:
    # - used_paths: impede repetir path (intra-render) e também aplica histórico (inter-episódio, por idioma)
    # - used_names: impede repetir o mesmo arquivo por nome (caso haja duplicatas/paths diferentes)
    used_paths: set[str] = set(used_paths_in) if used_paths_in else set()
    used_names: set[str] = set()

    def _is_avoided(tag: Optional[str]) -> bool:
        return bool(tag) and tag.lower() in avoid_lc

    def _safe_fallback() -> str:
        """
        Retorna um fallback que:
          - existe no inventário (clips_index)
          - NÃO está em avoid
        Ordem de preferência: fallback_tag -> secondary -> alguns padrões -> qualquer disponível
        """
        prefer = [
            fallback_tag,
            SECONDARY_FALLBACK_TAG,
            "business",
            "office_night",
            "city",
            "time",
            "reflection",
            "decision",
            "expenses",
            "data",
        ]
        for t in prefer:
            if t and (not _is_avoided(t)) and (clips_index.get(t) or []):
                return t

        for t, pool in clips_index.items():
            if pool and (not _is_avoided(t)):
                return t

        raise RuntimeError("avoid_tags removeu todas as tags disponíveis no inventário.")

    def pick_unique_clip(desired_tag: str) -> Tuple[str, Path]:
        """
        Resolve desired_tag -> (resolved_tag, clip_path) com:
          - sinônimos + fallback em cascata
          - anti-repeat global por arquivo e por nome
          - avoid como hard constraint (não fura via sinônimos/fallback)
        """
        safe_fb = _safe_fallback()

        candidates = [desired_tag]
        candidates += SIMILAR_TAGS.get(desired_tag, [])
        candidates += [SECONDARY_FALLBACK_TAG, safe_fb]

        # dedup + remove vazios + remove evitados
        cand2: List[str] = []
        for c in _dedup_keep_order([c for c in candidates if (c or "").strip()]):
            if _is_avoided(c):
                continue
            cand2.append(c)

        for t in cand2:
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

            # resolved_tag é "t"; requested_tag será setado pelo chamador
            return (t, chosen)

        if history_policy == "relax":
            # permite repetir (apenas como último recurso), MAS ainda respeita avoid
            for t in cand2:
                pool = clips_index.get(t) or []
                if pool:
                    chosen = rng.choice(pool)
                    used_paths.add(str(chosen.resolve()))
                    used_names.add(chosen.name.lower())
                    return (t, chosen)

            # salvage: qualquer tag com inventário que não esteja em avoid
            for t, pool in clips_index.items():
                if pool and (not _is_avoided(t)):
                    chosen = rng.choice(pool)
                    used_paths.add(str(chosen.resolve()))
                    used_names.add(chosen.name.lower())
                    return (t, chosen)

        raise RuntimeError(
            f"Sem clipes disponíveis respeitando avoid para '{desired_tag}' (nem sinônimos/fallbacks). "
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

                if _is_avoided(desired):
                    desired = _safe_fallback()

                resolved_tag, clip_path = pick_unique_clip(desired)
                scenes.append(
                    Scene(
                        tag=resolved_tag,
                        text=b.text,
                        duration=float(block_dur),
                        clip_path=clip_path,
                        forced_tags=None,
                        requested_tag=desired,
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

                if _is_avoided(t):
                    t = _safe_fallback()

                resolved_tag, clip_path = pick_unique_clip(t)
                scenes.append(
                    Scene(
                        tag=resolved_tag,
                        text=tx,
                        duration=float(dur),
                        clip_path=clip_path,
                        forced_tags=b.tags,
                        requested_tag=t,
                    )
                )
        else:
            # keywords mode: sempre 1 cena por bloco (desde que haja budget)
            desired = choose_tag(b.text, tag_order, tags, fallback=fallback_tag)

            # aplica avoid também no modo keywords (hard, sem voltar pro proibido)
            if _is_avoided(desired):
                desired = _safe_fallback()

            resolved_tag, clip_path = pick_unique_clip(desired)
            scenes.append(
                Scene(
                    tag=resolved_tag,
                    text=b.text,
                    duration=float(block_dur),
                    clip_path=clip_path,
                    forced_tags=None,
                    requested_tag=desired,
                )
            )

    # 3) Ajusta durações finais para bater o áudio e respeitar bounds
       # 3) Ajusta durações finais para bater o áudio e respeitar bounds
    if scenes:
        n_scenes = len(scenes)

        # por padrão, respeita o min_scene_eff
        fit_min = min_scene_eff
        fit_max = max_scene_eff

        # se tiver MUITA cena (ex.: 80, 100, 130...), relaxa o mínimo
        # para não achatar tudo em min_scene_eff
        if n_scenes > 60:
            fit_min = max(0.8, min_scene_eff * 0.6)  # ex.: 3.0 -> 1.8

        final_durs = fit_durations_with_bounds(
            [s.duration for s in scenes],
            total=audio_dur,
            minv=fit_min,
            maxv=fit_max,
        )
        for sc, d in zip(scenes, final_durs):
            sc.duration = float(d)

        # drift final
        drift = audio_dur - sum(s.duration for s in scenes)
        if abs(drift) > 0.02:
            scenes[-1].duration = max(0.3, scenes[-1].duration + drift)

    return scenes


def _merge_blocks(blocks: List[ScriptBlock], n_target: int) -> List[ScriptBlock]:
    if n_target <= 0:
        return []
    if len(blocks) <= n_target:
        return blocks[:]

    # agrupa adjacentes para reduzir quantidade
    # estratégia simples: distribui em n_target buckets contíguos
    buckets: List[List[ScriptBlock]] = [[] for _ in range(n_target)]
    # tamanho aproximado de cada bucket
    base = len(blocks) // n_target
    rem = len(blocks) % n_target

    idx = 0
    for i in range(n_target):
        take = base + (1 if i < rem else 0)
        buckets[i] = blocks[idx: idx + take]
        idx += take

    merged: List[ScriptBlock] = []
    for group in buckets:
        if not group:
            continue
        texts = []
        tags: List[str] = []
        for b in group:
            if b.text.strip():
                texts.append(b.text.strip())
            if b.tags:
                tags.extend([t for t in b.tags if t])
        tags = _dedup_keep_order(tags)
        merged.append(ScriptBlock(tags=tags if tags else None, text="\n\n".join(texts).strip()))
    return merged


def _split_blocks(blocks: List[ScriptBlock], n_target: int) -> List[ScriptBlock]:
    if n_target <= 0:
        return []
    if len(blocks) >= n_target:
        return blocks[:]

    out: List[ScriptBlock] = blocks[:]
    # enquanto faltar, divide o maior bloco (por frases/palavras via split_text_into_n_chunks)
    while len(out) < n_target:
        # escolhe o bloco com mais palavras
        i = max(range(len(out)), key=lambda k: word_count(out[k].text))
        b = out.pop(i)
        missing = n_target - len(out) + 1
        # divide em no máximo 2 por iteração (controlado, evita explosão)
        parts = split_text_into_n_chunks(b.text, 2 if missing > 1 else 1)
        parts = [p for p in parts if p.strip()]
        if len(parts) <= 1:
            # não deu para dividir, reintroduz e quebra
            out.insert(i, b)
            break
        # mantém tags iguais nas partes
        for j, tx in enumerate(parts):
            out.insert(i + j, ScriptBlock(tags=b.tags, text=tx))
    return out


def resample_blocks_to_n(blocks: List[ScriptBlock], n_target: int) -> List[ScriptBlock]:
    blocks = [b for b in blocks if (b.text or "").strip()]
    if not blocks:
        return []
    if n_target <= 0:
        return blocks
    if len(blocks) == n_target:
        return blocks
    if len(blocks) > n_target:
        return _merge_blocks(blocks, n_target)
    return _split_blocks(blocks, n_target)


# --------------------------
# Builder novo: SEQUÊNCIA (imagens OU vídeos)
# --------------------------
def build_scenes_sequence(
    blocks: List[ScriptBlock],
    audio_dur: float,
    sequence_files: List[Path],
    min_scene: float,
    max_scene: float,
    seed: int,
    tag_mode: str = "editor",   # mantém o mesmo contrato: editor|text
    seq_avg: Optional[float] = None,
    seq_min: Optional[float] = None,
    seq_max: Optional[float] = None,
    seq_jitter: float = 0.18,
    seq_max_scenes: int = 45,
) -> List[Scene]:
    """
    Igual ao build_scenes em termos de durações/expansão,
    mas o clip_path vem de sequence_files (1..N) em ordem, ciclando.

    Observação: aqui o "tag" vira apenas "sequence" (informativo).
    """
    rng = random.Random(seed)

    if not sequence_files:
        raise RuntimeError("sequence_files vazio (nada para usar no modo sequência).")

    idx_seq = 0

    def next_media() -> Path:
        nonlocal idx_seq
        p = sequence_files[idx_seq % len(sequence_files)]
        idx_seq += 1
        return p

    # se seq_avg foi fornecido, usa o modo "editorial por média"
    if seq_avg is not None and seq_avg > 0 and audio_dur > 0:
        eff_min = float(seq_min) if (seq_min is not None) else float(min_scene)
        eff_max = float(seq_max) if (seq_max is not None) else float(max_scene)
        eff_min = max(0.3, eff_min)
        eff_max = max(eff_min + 0.2, eff_max)

        n_target = int(round(audio_dur / float(seq_avg)))
        n_target = max(1, n_target)
        n_target = min(n_target, max(1, int(seq_max_scenes)))

        blocks2 = resample_blocks_to_n(blocks, n_target)
        if not blocks2:
            return []

        rng = random.Random(seed)

        # durações iniciais: média + jitter (evita metronomo)
        raw_durs: List[float] = []
        for _ in blocks2:
            j = float(seq_jitter)
            j = max(0.0, min(j, 0.60))
            d = float(seq_avg) * (1.0 + rng.uniform(-j, +j))
            raw_durs.append(d)

        # ajusta para somar audio_dur respeitando bounds
        final_durs = fit_durations_with_bounds(raw_durs, total=audio_dur, minv=eff_min, maxv=eff_max)

        scenes: List[Scene] = []
        for b, dur in zip(blocks2, final_durs):
            scenes.append(
                Scene(
                    tag="sequence",
                    text=b.text,
                    duration=float(dur),
                    clip_path=next_media(),
                    forced_tags=b.tags if b.tags else None,
                    requested_tag=None,
                )
            )

        drift = audio_dur - sum(s.duration for s in scenes)
        if scenes and abs(drift) > 0.02:
            scenes[-1].duration = max(0.3, scenes[-1].duration + drift)

        return scenes

    # 1) Duração base por BLOCO (proporcional a palavras)
    texts = [b.text for b in blocks]
    counts = [max(1, word_count(t)) for t in texts]
    total_words = sum(counts) or 1

    raw_block_durs = [(c / total_words) * audio_dur for c in counts]
    raw_block_durs = [d * rng.uniform(0.90, 1.10) for d in raw_block_durs]

    n_blocks = max(1, len(texts))
    avg = audio_dur / n_blocks if audio_dur > 0 else min_scene

    min_scene_eff = max(0.3, min(min_scene, avg * 0.90))
    max_scene_eff = max(max_scene, min_scene_eff + 0.2)

    block_durs = fit_durations_with_bounds(
        raw_block_durs,
        audio_dur,
        min_scene_eff,
        min(max_scene_eff, avg * 1.80),
    )

    # 1.1) Budget global de cenas (mesmo critério do build_scenes)
    min_for_budget = max(0.3, min_scene_eff)
    if audio_dur > 0 and min_for_budget > 0:
        raw_max_scenes = int(audio_dur // min_for_budget)
    else:
        raw_max_scenes = len(blocks) or 1
    max_scenes = max(1, raw_max_scenes)

    # 2) Expande blocos em cenas (consome N itens da sequência)
    scenes: List[Scene] = []
    total_blocks = len(blocks)

    for bi, (b, block_dur) in enumerate(zip(blocks, block_durs)):
        remaining_blocks = total_blocks - bi
        remaining_budget = max_scenes - len(scenes)

        if remaining_budget <= 0:
            break

        if b.tags:
            forced_all = select_forced_tags_for_block(b.tags)
            if forced_all:
                # respeita budget para não matar os blocos restantes
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
                    wcs = [max(1, word_count(x)) for x in chunks]
                    s = sum(wcs) or 1
                    durs = [(w / s) * float(block_dur) for w in wcs]

                for _t, tx, dur in zip(forced, chunks, durs):
                    scenes.append(
                        Scene(
                            tag="sequence",
                            text=tx,
                            duration=float(dur),
                            clip_path=next_media(),
                            forced_tags=b.tags,
                            requested_tag=None,
                        )
                    )
                continue

        # Sem tags úteis: 1 cena por bloco
        scenes.append(
            Scene(
                tag="sequence",
                text=b.text,
                duration=float(block_dur),
                clip_path=next_media(),
                forced_tags=b.tags if b.tags else None,
                requested_tag=None,
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

        drift = audio_dur - sum(s.duration for s in scenes)
        if abs(drift) > 0.02:
            scenes[-1].duration = max(0.3, scenes[-1].duration + drift)

    return scenes


def build_scenes_sequence_blocks(
    blocks: List[ScriptBlock],
    audio_dur: float,
    sequence_files: List[Path],
    seed: int,
) -> List[Scene]:
    """
    Modo 'por bloco':
      - 1 bloco de [SCRIPT_BLOCKS] = 1 cena
      - duração uniforme = audio_dur / n_blocks
      - clip_path = sequence_files[i % len(sequence_files)]
      - ignora min_scene/max_scene, word_count, tag_mode, etc.
    """
    if not blocks:
        return []

    if not sequence_files:
        raise RuntimeError("sequence_files vazio (nada para usar no modo sequência por bloco).")

    rng = random.Random(seed)  # só mantém interface, se quiser usar mais tarde

    n = len(blocks)
    if audio_dur > 0:
        base_dur = float(audio_dur) / float(n)
    else:
        # fallback se por algum motivo audio_dur estiver 0 ou negativo
        base_dur = 5.0

    scenes: List[Scene] = []
    for i, b in enumerate(blocks):
        clip = sequence_files[i % len(sequence_files)]
        scenes.append(
            Scene(
                tag="sequence_block",   # só informativo para log/debug
                text=b.text,
                duration=base_dur,
                clip_path=clip,
                forced_tags=b.tags if b.tags else None,
                requested_tag=None,
            )
        )

    # Ajuste fino de drift por conta de arredondamento
    total = sum(s.duration for s in scenes)
    drift = audio_dur - total
    if scenes and abs(drift) > 0.02:
        scenes[-1].duration = max(0.3, scenes[-1].duration + drift)

    return scenes
