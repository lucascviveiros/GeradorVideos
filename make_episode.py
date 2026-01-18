#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from typing import Set, Dict, Any, Optional

import argparse
from pathlib import Path

from episode_plan import (
    build_scenes,
    build_scenes_sequence,      # <-- NOVO
    index_sequence_media,       # <-- NOVO
    load_tags_yaml,
    parse_script_blocks,
)

from episode_render import render_video

try:
    from moviepy import AudioFileClip
except Exception:
    from moviepy.editor import AudioFileClip


# ANSI (cores no terminal)
ANSI_RED = "\033[91m"
ANSI_BLUE = "\033[94m"
ANSI_GREEN = "\033[92m"
ANSI_RESET = "\033[0m"


def resolve_path(user_path: str, must_exist: bool = False, is_dir: bool = False) -> Path:
    p = Path(user_path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve(strict=False)

    if must_exist:
        if is_dir and not p.is_dir():
            raise RuntimeError(f"Pasta não encontrada: {p}")
        if not is_dir and not p.is_file():
            raise RuntimeError(f"Arquivo não encontrado: {p}")
    return p


def _normalize_key(p: Path) -> str:
    # Sempre grava/consulta com path absoluto resolvido (evita duplicidade por relativo diferente)
    return str(p.resolve())


def _infer_episode_id_from_audio(audio_path: Path) -> str:
    # Heurística simples e estável: pega "ep###" do nome do arquivo (ex: ep002_pt.mp3 -> ep002)
    name = audio_path.stem.lower()
    import re
    m = re.search(r"(ep\d+)", name)
    if m:
        return m.group(1)
    return audio_path.stem


def _history_get_lang_bucket(data: Dict[str, Any], lang: str) -> Dict[str, Any]:
    if "langs" not in data or not isinstance(data.get("langs"), dict):
        data["langs"] = {}
    if lang not in data["langs"] or not isinstance(data["langs"].get(lang), dict):
        data["langs"][lang] = {"clips": {}}
    if "clips" not in data["langs"][lang] or not isinstance(data["langs"][lang].get("clips"), dict):
        data["langs"][lang]["clips"] = {}
    return data["langs"][lang]


def load_history(path: Path, reset: bool) -> Dict[str, Any]:
    if reset:
        return {"langs": {}}
    if not path.exists():
        return {"langs": {}}
    try:
        # utf-8-sig: tolera BOM do VS Code/Windows (evita JSONDecodeError em arquivos "válidos")
        raw = path.read_text(encoding="utf-8-sig")
        if not raw.strip():
            return {"langs": {}}
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {"langs": {}}
        # garante estrutura mínima
        if "langs" not in data or not isinstance(data.get("langs"), dict):
            data["langs"] = {}
        return data
    except Exception:
        raise RuntimeError(f"history_file inválido/corrompido: {path}")


def save_history(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _record_scenes_in_history(history: Dict[str, Any], lang: str, episode_id: str, scenes) -> None:
    bucket = _history_get_lang_bucket(history, lang)
    clips = bucket["clips"]
    for sc in scenes:
        k = _normalize_key(sc.clip_path)
        # registra o episódio onde apareceu pela primeira vez (não sobrescreve)
        if k not in clips:
            clips[k] = episode_id


def _warn_reuse(lang: str, episode_id: str, clip_path: Path, prev_episode: Optional[str]) -> None:
    prev = prev_episode or "desconhecido"
    msg = (
        f'[WARN][{lang}][{episode_id}] REUTILIZANDO CLIPE: "{clip_path}" '
        f'(já usado no episódio {prev})'
    )
    print(f"{ANSI_RED}{msg}{ANSI_RESET}")


def _info_synonym(lang: str, episode_id: str, desired_tag: str, chosen_tag: str, clip_path: Path) -> None:
    msg = (
        f'[INFO][{lang}][{episode_id}] usando sinônimo: desired="{desired_tag}" '
        f'-> chosen_tag="{chosen_tag}" | clip="{clip_path.name}"'
    )
    print(f"{ANSI_BLUE}{msg}{ANSI_RESET}")


def _info_first_time(lang: str, episode_id: str, tag: str, clip_path: Path) -> None:
    msg = (
        f'[OK][{lang}][{episode_id}] clipe novo: '
        f'tag="{tag}" | clip="{clip_path.name}"'
    )
    print(f"{ANSI_GREEN}{msg}{ANSI_RESET}")


def _print_plan_colored(lang: str, scenes, prev_map_lang: Dict[str, str]) -> None:
    """
    ÚNICO output desejado (quando --print_plan estiver ligado):
      - Formato PLAN (uma linha por cena)
      - Cor por linha:
          vermelho = reutilizando (já existe no history)
          azul     = sinônimo (SYN)
          verde    = clipe novo
      - Sem logs soltos [INFO]/[OK]/[WARN] fora do PLAN
    """
    print(f"\n--- PLAN ({lang}) ---")

    t0 = 0.0
    for i, sc in enumerate(scenes, 1):
        _ = t0 + float(sc.duration)  # mantido apenas para não mexer na estrutura de cálculo

        desired = getattr(sc, "requested_tag", None)
        chosen = getattr(sc, "tag", None)

        # Coluna TAG: mostra "desired -> chosen" quando houve sinônimo
        if desired and chosen and desired != chosen:
            tag_col = f"{desired} -> {chosen}"
            syn_mark = "SYN"
        else:
            tag_col = f"{chosen}"
            syn_mark = "   "

        k = _normalize_key(sc.clip_path)
        reused = k in prev_map_lang
        prev_ep = prev_map_lang.get(k)

        line = f"{i:02d} | {sc.duration:5.2f}s | {syn_mark} | {tag_col:<28} | {sc.clip_path.name}"

        # Texto extra opcional para reuse (mantém o formato e só anexa um sufixo)
        if reused:
            prev_txt = prev_ep or "desconhecido"
            line += f" (reutilizando: {prev_txt})"

        # Precedência: reutilizando (vermelho) > sinônimo (azul) > novo (verde)
        if reused:
            color = ANSI_RED
        elif syn_mark == "SYN":
            color = ANSI_BLUE
        else:
            color = ANSI_GREEN

        print(f"{color}{line}{ANSI_RESET}")

        t0 = t0 + float(sc.duration)


def load_avoid_tags(avoid_path: Path) -> Set[str]:
    """
    avoid.json deve ser uma LISTA de tags/temas a evitar.
    Ex: ["matrix", "cars", "politics"]
    """
    if not avoid_path.exists():
        return set()
    data = json.loads(avoid_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise RuntimeError(f"avoid.json deve ser uma LISTA, ex: ['matrix','cars']. Arquivo: {avoid_path}")
    return {str(x).strip().lower() for x in data if str(x).strip()}


def _is_avoided(path: Path, avoid_folders: Optional[Set[str]]) -> bool:
    if not avoid_folders:
        return False
    parts = [p.lower() for p in path.parts]
    return any(a in parts for a in avoid_folders)


def main():
    ap = argparse.ArgumentParser(description="Auto-montagem: roteiro + áudio + clipes por tag.")
    ap.add_argument(
        "--avoid",
        action="store_true",
        help="Se ligado, ignora pastas definidas em um JSON por canal (ex.: matrix, cars...).",
    )

    ap.add_argument(
        "--history_file",
        default=None,
        help="JSON de histórico de clipes usados na série. Ex: decisao_invisivel/history.json",
    )
    ap.add_argument(
        "--history_policy",
        default="strict",
        choices=["strict", "relax"],
        help="strict=erro se não houver clipes novos; relax=permite repetir se acabar inventário",
    )
    ap.add_argument(
        "--history_reset",
        action="store_true",
        help="Se ligado, ignora histórico existente e sobrescreve a partir deste episódio",
    )

    # NOVO: múltiplos canais (histórico por canal)
    ap.add_argument(
        "--channel_name",
        default=None,
        help="Nome do canal (pasta). Ex: decisao_invisivel. Se fornecido e --history_file não for passado, usa {history_root}/{channel_name}/history.json",
    )
    ap.add_argument(
        "--history_root",
        default=".",
        help="Pasta base onde ficam os canais (default: .). Ex: ./channels",
    )

    ap.add_argument("--audio", required=True)
    ap.add_argument("--text", required=False)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--tags", default="tags.yaml")

    ap.add_argument("--langs", default="pt,en,es")
    ap.add_argument("--audio_pattern", default=None, help="Ex: audio/ep001_{lang}.mp3")
    ap.add_argument("--out_pattern", default=None, help="Ex: out/ep001_{lang}.mp4")
    ap.add_argument("--out", default="out/episode.mp4")

    ap.add_argument("--fallback", default="abstract_dark")
    ap.add_argument("--min_scene", type=float, default=1.8)
    ap.add_argument("--max_scene", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--tag_mode", default="editor", choices=["editor", "text"])
    ap.add_argument("--print_plan", action="store_true")

    # NOVO: apenas plano (sem render), mas salvando no history
    ap.add_argument(
        "--plan_only",
        action="store_true",
        help="Se ligado, imprime o PLAN e NÃO renderiza; ainda assim registra clipes no history (se configurado).",
    )

    # render
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--vcodec", default="auto")
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--nvenc_preset", default="hq")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--bitrate", default=None)
    ap.add_argument("--audio_bitrate", default="192k")
    ap.add_argument("--debug_ffmpeg", action="store_true")

    # NOVO: render de teste rápido (qualidade ruim)
    ap.add_argument(
        "--teste_render",
        action="store_true",
        help="Se ligado, renderiza em qualidade baixa para teste (mais rápido).",
    )

    # cache
    ap.add_argument("--cache_clips", action="store_true")
    ap.add_argument("--cache_dir", default="clips_cache")
    ap.add_argument("--cache_force", action="store_true")
    ap.add_argument("--cache_fps", type=int, default=30)

    # sequência: controle editorial por "segundos médios por imagem"
    ap.add_argument(
        "--seq_avg",
        type=float,
        default=None,
        help="(sequência) duração média alvo por cena (ex: 6.0). Se None, usa o modo antigo.",
    )
    ap.add_argument(
        "--seq_min",
        type=float,
        default=None,
        help="(sequência) mínimo por cena. Se None, cai em --min_scene.",
    )
    ap.add_argument(
        "--seq_max",
        type=float,
        default=None,
        help="(sequência) máximo por cena. Se None, cai em --max_scene.",
    )
    ap.add_argument(
        "--seq_jitter",
        type=float,
       default=0.18,
        help="(sequência) variação relativa em torno de seq_avg (0.18 = ±18%%).",
    )
    ap.add_argument(
        "--seq Reid_max_scenes",
        type=int,
        default=45,
        help="(sequência) teto duro de cenas (evita 90+ cortes).",
    )

    # make_episode.py (dentro do main())
    mx = ap.add_mutually_exclusive_group()
    mx.add_argument("--use_sequence_image", default=None, help="Pasta com 1.png,2.jpg,... para usar como sequência")
    mx.add_argument("--use_sequence_video", default=None, help="Pasta com 1.mp4,2.mp4,... para usar como sequência")

    args = ap.parse_args()

    audio_p = resolve_path(args.audio, must_exist=True, is_dir=False)

    # Mantém --text como optional na CLI, mas evita crash silencioso se omitido
    if not args.text:
        raise RuntimeError("--text é obrigatório (arquivo do roteiro/estrutura).")
    text_p = resolve_path(args.text, must_exist=True, is_dir=False)

    clips_p = resolve_path(args.clips, must_exist=True, is_dir=True)
    tags_p = resolve_path(args.tags, must_exist=True, is_dir=False)

    tag_order, tags = load_tags_yaml(tags_p)
    blocks = parse_script_blocks(text_p)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    if not langs:
        raise RuntimeError("--langs inválido. Ex: pt,en,es")

    def resolve_audio_for_lang(lang: str) -> Path:
        if args.audio_pattern:
            return resolve_path(args.audio_pattern.format(lang=lang), must_exist=True, is_dir=False)

        p = audio_p
        name = p.stem
        suf = p.suffix or ".mp3"
        for known in ("_pt", "_en", "_es"):
            if name.endswith(known):
                return resolve_path(
                    str(p.with_name(name.replace(known, f"_{lang}") + suf)),
                    must_exist=True,
                    is_dir=False,
                )
        return resolve_path(str(p.with_name(f"{name}_{lang}{suf}")), must_exist=True, is_dir=False)

    def resolve_out_for_lang(lang: str) -> Path:
        if args.out_pattern:
            return resolve_path(args.out_pattern.format(lang=lang), must_exist=False, is_dir=False)
        p = Path(args.out)
        return resolve_path(str(p.with_name(f"{p.stem}_{lang}{p.suffix}")), must_exist=False, is_dir=False)

    cache_dir_p = resolve_path(args.cache_dir, must_exist=False, is_dir=True)

    # histórico (persistente entre episódios) - por idioma, registrando episódio de origem
    history_p = None
    history_data: Dict[str, Any] = {"langs": {}}

    # Resolve automático do histórico por canal:
    # prioridade: --history_file (manual) > --channel_name (auto) > nenhum histórico
    if args.history_file:
        history_p = resolve_path(args.history_file, must_exist=False, is_dir=False)
    elif args.channel_name:
        root = resolve_path(args.history_root, must_exist=False, is_dir=True)
        history_p = resolve_path(str(root / args.channel_name / "history.json"), must_exist=False, is_dir=False)

    if history_p is not None:
        history_data = load_history(history_p, args.history_reset)


    # --- avoid tags (por canal): ./<channel_name>/avoid.json ---
    avoid_tags: Set[str] = set()

    if args.avoid:
        if not args.channel_name:
            raise RuntimeError("--avoid requer --channel_name (para resolver ./<channel_name>/avoid.json)")

        # mesmo root/canal do history.json (ao lado do history.json)
        root = resolve_path(args.history_root, must_exist=False, is_dir=True)
        avoid_p = resolve_path(str(root / args.channel_name / "avoid.json"), must_exist=False, is_dir=False)

        avoid_tags = load_avoid_tags(avoid_p)

        if avoid_tags:
            print(f"[INFO] --avoid ligado: evitando tags/temas (canal={args.channel_name}): {sorted(avoid_tags)}")
        else:
            # Se você preferir falhar aqui, troque por RuntimeError(...)
            print(f"[INFO] --avoid ligado, mas {avoid_p} não existe ou está vazio.")



    for lang in langs:
        audio_lang = resolve_audio_for_lang(lang)
        out_lang = resolve_out_for_lang(lang)

        episode_id = _infer_episode_id_from_audio(audio_lang)

        a = AudioFileClip(str(audio_lang))
        audio_dur = a.duration
        a.close()

        # conjunto de clipes já usados na série PARA ESTE IDIOMA
        used_series_lang: Set[str] = set()
        prev_map_lang: Dict[str, str] = {}
        if history_p is not None:
            bucket = _history_get_lang_bucket(history_data, lang)
            clips_map = bucket.get("clips", {})
            if isinstance(clips_map, dict):
                prev_map_lang = {str(k): str(v) for k, v in clips_map.items()}
                used_series_lang = set(prev_map_lang.keys())

        if args.use_sequence_image:
            seq_dir = resolve_path(args.use_sequence_image, must_exist=True, is_dir=True)
            seq_files = index_sequence_media(seq_dir, mode="image")

            # aplica histórico por idioma também no modo sequência
            if history_p is not None:
                filtered = [p for p in seq_files if _normalize_key(p) not in used_series_lang]
                if filtered:
                    seq_files = filtered
                elif args.history_policy == "strict":
                    raise RuntimeError(
                        f"Sem mídia nova em sequência (image) para lang={lang} após aplicar histórico: {history_p}. "
                        f"Use --history_policy relax ou abasteça a pasta {seq_dir}."
                    )

            scenes = build_scenes_sequence(
                blocks=blocks,
                audio_dur=audio_dur,
                sequence_files=seq_files,
                min_scene=args.min_scene,
                max_scene=args.max_scene,
                seed=args.seed,
                tag_mode=args.tag_mode,
                seq_avg=args.seq_avg,
                seq_min=args.seq_min,
                seq_max=args.seq_max,
                seq_jitter=args.seq_jitter,
                seq_max_scenes=getattr(args, "seq_max_scenes", 45),
            )

        elif args.use_sequence_video:
            seq_dir = resolve_path(args.use_sequence_video, must_exist=True, is_dir=True)
            seq_files = index_sequence_media(seq_dir, mode="video")

            # aplica histórico por idioma também no modo sequência
            if history_p is not None:
                filtered = [p for p in seq_files if _normalize_key(p) not in used_series_lang]
                if filtered:
                    seq_files = filtered
                elif args.history_policy == "strict":
                    raise RuntimeError(
                        f"Sem mídia nova em sequência (video) para lang={lang} após aplicar histórico: {history_p}. "
                        f"Use --history_policy relax ou abasteça a pasta {seq_dir}."
                    )

            scenes = build_scenes_sequence(
                blocks=blocks,
                audio_dur=audio_dur,
                sequence_files=seq_files,
                min_scene=args.min_scene,
                max_scene=args.max_scene,
                seed=args.seed,
                tag_mode=args.tag_mode,
            )

        else:
            # modo TAGS: histórico por idioma via build_scenes
            scenes = build_scenes(
                blocks=blocks,
                audio_dur=audio_dur,
                tag_order=tag_order,
                tags=tags,
                clips_root=clips_p,
                fallback_tag=args.fallback,
                min_scene=args.min_scene,
                max_scene=args.max_scene,
                seed=args.seed,
                tag_mode=args.tag_mode,
                used_paths_in=used_series_lang if history_p is not None else None,
                history_policy=args.history_policy,
                avoid_tags=avoid_tags if args.avoid else None,  # <<< LINHA NOVA
            )

        # Quando --print_plan estiver ligado:
        # - imprime SOMENTE o PLAN colorido (sem logs soltos)
        # - NÃO interrompe render (remove o continue)
        # Quando --plan_only estiver ligado:
        # - imprime o PLAN (mesmo que --print_plan não esteja ligado)
        # - NÃO renderiza
        # - SALVA no history
        if args.print_plan or args.plan_only:
            _print_plan_colored(lang, scenes, prev_map_lang)

        if args.plan_only:
            # atualiza histórico mesmo sem render (por idioma, registrando episódio onde apareceu)
            if history_p is not None:
                _record_scenes_in_history(history_data, lang, episode_id, scenes)
                save_history(history_p, history_data)
            print(f"OK ({lang}): PLAN_ONLY (sem render): {out_lang}")
            continue

        # --- OVERRIDES de TESTE (qualidade ruim e rápido) ---
        target_w = 1920
        target_h = 1080
        fast_mode = False

        fps = args.fps
        vcodec = args.vcodec
        preset = args.preset
        nvenc_preset = args.nvenc_preset
        crf = args.crf
        bitrate = args.bitrate
        audio_bitrate = args.audio_bitrate
        test_render = False
        test_scale = 720
        test_crf = 35
        test_preset = "ultrafast"
        test_audio_bitrate = "96k"

        if args.teste_render:
            print("[INFO] --teste_render ligado: render rápido (scale=480 / fps=24 / libx264 ultrafast / crf=35 / audio=96k)")
            fps = 24

            test_render = True
            test_scale = 480          # altura do vídeo (480p)
            test_crf = 35
            test_preset = "ultrafast"
            test_audio_bitrate = "96k"

        render_video(
            scenes=scenes,
            audio_path=audio_lang,
            out_path=out_lang,
            fps=fps,
            vcodec=args.vcodec,
            preset=args.preset,
            nvenc_preset=args.nvenc_preset,
            crf=args.crf,
            bitrate=args.bitrate,
            audio_bitrate=args.audio_bitrate,
            debug_ffmpeg=args.debug_ffmpeg,
            cache_clips=args.cache_clips,
            cache_dir=cache_dir_p,
            cache_force=args.cache_force,
            cache_fps=args.cache_fps,
            test_render=test_render,
            test_scale=test_scale,
            test_crf=test_crf,
            test_preset=test_preset,
            test_audio_bitrate=test_audio_bitrate,
        )

        # atualiza histórico após render (por idioma, registrando episódio onde apareceu)
        if history_p is not None:
            _record_scenes_in_history(history_data, lang, episode_id, scenes)
            save_history(history_p, history_data)

        print(f"OK ({lang}): {out_lang}")


if __name__ == "__main__":
    main()
