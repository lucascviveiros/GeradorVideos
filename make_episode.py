#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from episode_plan import (
    build_scenes,
    load_tags_yaml,
    parse_script_blocks,
)
from episode_render import render_video

try:
    from moviepy import AudioFileClip
except Exception:
    from moviepy.editor import AudioFileClip


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


def main():
    ap = argparse.ArgumentParser(description="Auto-montagem: roteiro + áudio + clipes por tag.")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--text", required=True)
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

    # render
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--vcodec", default="auto")
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--nvenc_preset", default="hq")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--bitrate", default=None)
    ap.add_argument("--audio_bitrate", default="192k")
    ap.add_argument("--debug_ffmpeg", action="store_true")

    # cache
    ap.add_argument("--cache_clips", action="store_true")
    ap.add_argument("--cache_dir", default="clips_cache")
    ap.add_argument("--cache_force", action="store_true")
    ap.add_argument("--cache_fps", type=int, default=30)

    args = ap.parse_args()

    audio_p = resolve_path(args.audio, must_exist=True, is_dir=False)
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
                return resolve_path(str(p.with_name(name.replace(known, f"_{lang}") + suf)), must_exist=True, is_dir=False)
        return resolve_path(str(p.with_name(f"{name}_{lang}{suf}")), must_exist=True, is_dir=False)

    def resolve_out_for_lang(lang: str) -> Path:
        if args.out_pattern:
            return resolve_path(args.out_pattern.format(lang=lang), must_exist=False, is_dir=False)
        p = Path(args.out)
        return resolve_path(str(p.with_name(f"{p.stem}_{lang}{p.suffix}")), must_exist=False, is_dir=False)

    cache_dir_p = resolve_path(args.cache_dir, must_exist=False, is_dir=True)

    for lang in langs:
        audio_lang = resolve_audio_for_lang(lang)
        out_lang = resolve_out_for_lang(lang)

        a = AudioFileClip(str(audio_lang))
        audio_dur = a.duration
        a.close()

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
        )

        if args.print_plan:
            print(f"\n--- PLAN ({lang}) ---")
            for i, sc in enumerate(scenes, 1):
                print(f"{i:02d} | {sc.duration:.2f}s | {sc.tag} | {sc.clip_path.name}")
            continue

        render_video(
            scenes=scenes,
            audio_path=audio_lang,
            out_path=out_lang,
            fps=args.fps,
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
        )

        print(f"OK ({lang}): {out_lang}")


if __name__ == "__main__":
    main()
