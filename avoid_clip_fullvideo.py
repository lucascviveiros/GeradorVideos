#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class PickResult:
    path: Path
    reused: bool           # True quando precisou repetir (não havia alternativa global)
    reason: str            # texto curto para debug


class ClipPicker:
    """
    Controla seleção de clipes evitando repetição no vídeo inteiro.
    Estratégia:
      - Mantém "shuffle bag" por tag (ordem embaralhada estável por seed).
      - Antes de repetir, tenta achar algum clipe dessa tag ainda não usado globalmente.
      - Se todos já foram usados, permite reuso (inevitável).
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.used_global: Set[str] = set()  # guarda str(path.resolve()) para estabilidade
        self.bags: Dict[str, List[Path]] = {}
        self.bag_pos: Dict[str, int] = {}

    def _key(self, p: Path) -> str:
        try:
            return str(p.resolve())
        except Exception:
            return str(p)

    def reset(self) -> None:
        self.used_global.clear()
        self.bags.clear()
        self.bag_pos.clear()

    def prime_tag(self, tag: str, clips: List[Path]) -> None:
        """
        Cria/atualiza o shuffle bag para a tag.
        Se já existir bag, não recria para manter consistência.
        """
        if tag in self.bags:
            return
        arr = list(clips)
        self.rng.shuffle(arr)
        self.bags[tag] = arr
        self.bag_pos[tag] = 0

    def pick(self, tag: str, clips: List[Path]) -> PickResult:
        if not clips:
            raise RuntimeError(f"ClipPicker.pick: lista vazia para tag '{tag}'")

        self.prime_tag(tag, clips)

        bag = self.bags[tag]
        n = len(bag)
        start = self.bag_pos[tag] % n

        # 1) tenta achar um ainda não usado globalmente (varre no máximo 1 volta)
        for step in range(n):
            idx = (start + step) % n
            cand = bag[idx]
            k = self._key(cand)
            if k not in self.used_global:
                self.used_global.add(k)
                self.bag_pos[tag] = (idx + 1) % n
                return PickResult(path=cand, reused=False, reason="picked_unused_global")

        # 2) se todos já foram usados, reusa (inevitável)
        cand = bag[start]
        self.bag_pos[tag] = (start + 1) % n
        return PickResult(path=cand, reused=True, reason="reused_all_used_global")
