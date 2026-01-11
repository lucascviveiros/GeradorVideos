#!/usr/bin/env python3
"""
Bulk stock video downloader via official APIs (Pixabay + optional Pexels).

- Respects basic rate limiting (configurable).
- Saves files with a predictable naming scheme.
- You must comply with each provider's license/terms and avoid abusive use.
Pixabay API docs: https://pixabay.com/api/docs/  (see citations in chat)
Pexels API docs: https://www.pexels.com/api/documentation/
"""

import os
import re
import time
import json
import math
import argparse
from pathlib import Path
from urllib.parse import urlencode

import requests

#py -3.10 .\download_videos.py --use-pexels --out video_pack --queries "luxury,hotel,night city" --per-query 20 --per-page 15 --sleep 1.2 --pexels-max-req-hour 180

#Pixabay
#py -3.10 .\download_videos.py --out video_pack --queries "luxury,hotel,night city" --per-query 20 --per-page 15 --sleep 1.2

#Pixabay + Pexels
#py -3.10 .\download_videos.py --use-pexels --out video_pack --queries "luxury,hotel,night city" --per-query 20 --per-page 15 --sleep 1.2 --pexels-max-req-hour 180


# ---- API KEYS (prefer ENV; fallback only for local testing) ----
# NOTE: Do NOT hardcode real keys in code committed/shared. Use env vars.
PIXABAY_KEY = os.getenv("PIXABAY_KEY") or "21575746-729803702f41aa842bf088c30"
PEXELS_KEY  = os.getenv("PEXELS_KEY")  or ""  # opcional


def safe_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s[:80] if s else "query"

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def download_file(url: str, out_path: Path, timeout=60, retries: int = 3) -> bool:
    """
    Returns True if downloaded now, False if skipped because file already exists.
    Retries on transient network errors and cleans up partial .part files.
    """
    if out_path.exists() and out_path.stat().st_size > 0:
        return False  # already present -> skip

    tmp = out_path.with_suffix(out_path.suffix + ".part")

    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.replace(out_path)
            return True
        except Exception:
            # remove partial
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)  # backoff: 1s, 2s, 4s

    return False


def pixabay_search_videos(api_key: str, query: str, page: int, per_page: int):
    # Pixabay video endpoint
    base = "https://pixabay.com/api/videos/"
    params = {
        "key": api_key,
        "q": query,
        "page": page,
        "per_page": per_page,
        "safesearch": "true",
    }
    url = base + "?" + urlencode(params)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def pick_pixabay_video_url(hit: dict) -> str | None:
    # Prefer larger files if available
    vids = hit.get("videos", {}) or {}
    for k in ("large", "medium", "small", "tiny"):
        v = vids.get(k)
        if isinstance(v, dict) and v.get("url"):
            return v["url"]
    return None


def existing_ids_in_dir(q_dir: Path, suffix: str) -> set[str]:
    """
    suffix example: '_pixabay_' or '_pexels_'
    extracts ids from filenames like: <query>_pixabay_<id>.mp4
    """
    ids = set()
    pat = re.compile(re.escape(suffix) + r"(\d+)\.mp4$", re.IGNORECASE)
    for f in q_dir.glob("*.mp4"):
        m = pat.search(f.name)
        if m:
            ids.add(m.group(1))
    return ids


def bulk_pixabay(api_key: str, queries: list[str], out_dir: Path,
                 per_query: int, per_page: int, sleep_s: float):
    ensure_dir(out_dir)
    for q in queries:
        q_dir = out_dir / f"pixabay_{safe_name(q)}"
        ensure_dir(q_dir)

        # avoid downloading videos that already exist in folder (by id)
        existing = existing_ids_in_dir(q_dir, "_pixabay_")

        downloaded = 0
        page = 1
        while downloaded < per_query:
            data = pixabay_search_videos(api_key, q, page=page, per_page=per_page)
            hits = data.get("hits", []) or []
            if not hits:
                break

            for hit in hits:
                if downloaded >= per_query:
                    break
                url = pick_pixabay_video_url(hit)
                if not url:
                    continue

                vid_id = str(hit.get("id", "na"))
                if vid_id in existing:
                    continue

                out_path = q_dir / f"{safe_name(q)}_pixabay_{vid_id}.mp4"
                try:
                    did = download_file(url, out_path)
                    if did:
                        downloaded += 1
                        existing.add(vid_id)
                        print(f"[Pixabay] {q}: {downloaded}/{per_query} -> {out_path.name}")
                    else:
                        # If a file exists but id wasn't parsed before, ensure we don't retry it
                        if out_path.exists() and out_path.stat().st_size > 0:
                            existing.add(vid_id)
                        # opcional: não printar para não poluir o log
                        pass
                except Exception as e:
                    print(f"[Pixabay] failed {q} id={vid_id}: {e}")

            page += 1
            time.sleep(sleep_s)

def pexels_search_videos(api_key: str, query: str, page: int, per_page: int):
    base = "https://api.pexels.com/videos/search"
    headers = {"Authorization": api_key}
    params = {"query": query, "page": page, "per_page": per_page}
    r = requests.get(base, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def pick_pexels_video_url(video: dict) -> str | None:
    # Choose best quality mp4 from video_files
    files = video.get("video_files", []) or []
    mp4s = [f for f in files if (f.get("file_type") or "").lower() == "video/mp4" and f.get("link")]
    if not mp4s:
        return None
    # Prefer higher width, then size
    mp4s.sort(key=lambda f: (f.get("width") or 0, f.get("file_size") or 0), reverse=True)
    return mp4s[0]["link"]

def bulk_pexels(api_key: str, queries: list[str], out_dir: Path,
                per_query: int, per_page: int, sleep_s: float, max_requests_per_hour: int):
    ensure_dir(out_dir)
    # Simple hourly limiter
    window_start = time.time()
    req_count = 0

    for q in queries:
        q_dir = out_dir / f"pexels_{safe_name(q)}"
        ensure_dir(q_dir)

        # avoid downloading videos that already exist in folder (by id)
        existing = existing_ids_in_dir(q_dir, "_pexels_")

        downloaded = 0
        page = 1
        while downloaded < per_query:
            # hourly window
            now = time.time()
            if now - window_start >= 3600:
                window_start = now
                req_count = 0
            if req_count >= max_requests_per_hour:
                sleep_left = 3600 - (now - window_start)
                sleep_left = max(0, sleep_left)
                print(f"[Pexels] rate cap reached, sleeping {int(sleep_left)}s")
                time.sleep(sleep_left + 1)
                continue

            data = pexels_search_videos(api_key, q, page=page, per_page=per_page)
            req_count += 1

            videos = data.get("videos", []) or []
            if not videos:
                break

            for v in videos:
                if downloaded >= per_query:
                    break
                url = pick_pexels_video_url(v)
                if not url:
                    continue

                vid_id = str(v.get("id", "na"))
                if vid_id in existing:
                    continue

                out_path = q_dir / f"{safe_name(q)}_pexels_{vid_id}.mp4"
                try:
                    did = download_file(url, out_path)
                    if did:
                        downloaded += 1
                        existing.add(vid_id)
                        print(f"[Pexels] {q}: {downloaded}/{per_query} -> {out_path.name}")
                    else:
                        if out_path.exists() and out_path.stat().st_size > 0:
                            existing.add(vid_id)
                        # opcional: não printar para não poluir o log
                        pass
                except Exception as e:
                    print(f"[Pexels] failed {q} id={vid_id}: {e}")

            page += 1
            time.sleep(sleep_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="video_pack", help="Output folder")
    ap.add_argument("--queries", default="nature,city,abstract,tech,business,food,travel,fitness,people,night",
                    help="Comma-separated search queries")
    ap.add_argument("--per-query", type=int, default=50, help="How many videos per query")
    ap.add_argument("--per-page", type=int, default=20, help="API page size")
    ap.add_argument("--sleep", type=float, default=0.8, help="Sleep seconds between API pages")

    #KEYS
    ap.add_argument("--pixabay-key", default=PIXABAY_KEY, help="Pixabay API key (or env PIXABAY_KEY)")
    ap.add_argument("--pexels-key", default=PEXELS_KEY, help="Pexels API key (or env PEXELS_KEY)")

    ap.add_argument("--use-pexels", action="store_true", help="Also download from Pexels")
    ap.add_argument("--pexels-max-req-hour", type=int, default=180, help="Conservative Pexels requests/hour")
    args = ap.parse_args()

    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    out_dir = Path(args.out)

    if not args.pixabay_key:
        print("Missing Pixabay key. Create one at pixabay.com and pass --pixabay-key or set PIXABAY_KEY.")
    else:
        bulk_pixabay(args.pixabay_key, queries, out_dir, args.per_query, args.per_page, args.sleep)

    if args.use_pexels:
        if not args.pexels_key:
            print("Missing Pexels key. Create one at pexels.com and pass --pexels-key or set PEXELS_KEY.")
        else:
            bulk_pexels(args.pexels_key, queries, out_dir, args.per_query, args.per_page,
                        args.sleep, args.pexels_max_req_hour)

if __name__ == "__main__":
    main()
