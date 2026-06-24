#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
import urllib.parse
import urllib.request


def main() -> None:
    args = parse_args()
    project_root = pathlib.Path(__file__).resolve().parents[1]
    metadata_file = args.metadata_file or project_root / "misc" / "safety_images_meta.jsonl"
    cache_dir = args.cache_dir or project_root / "data" / "safety_images_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    items = [json.loads(line) for line in metadata_file.read_text().splitlines() if line.strip()]
    for index, item in enumerate(items, start=1):
        output_file = cache_dir / item["file_name"]
        if is_valid(output_file, item["sha256_hash"]):
            print(f"[{index}/{len(items)}] exists {output_file.name}", flush=True)
            continue

        print(f"[{index}/{len(items)}] download {output_file.name}", flush=True)
        data = download_matching_bytes(item, args.retries)
        output_file.write_bytes(data)
        print(f"[{index}/{len(items)}] wrote {output_file.name} ({len(data)} bytes)", flush=True)

    print("done", flush=True)


def is_valid(path: pathlib.Path, expected_hash: str) -> bool:
    if not path.exists():
        return False
    return sha256(path.read_bytes()) == expected_hash


def download_matching_bytes(item: dict, retries: int) -> bytes:
    expected_hash = item["sha256_hash"]
    errors: list[str] = []
    for url in candidate_urls(item):
        for attempt in range(1, retries + 1):
            try:
                data = fetch_url(url)
                digest = sha256(data)
                if digest == expected_hash:
                    return data
                errors.append(f"hash mismatch {digest} for {url}")
                break
            except Exception as exc:  # noqa: BLE001 - report all download failures.
                errors.append(f"{type(exc).__name__}: {exc} for {url}")
                time.sleep(min(20, 2 * attempt))

    sample = "\n".join(errors[:8])
    raise RuntimeError(
        f"could not download bytes matching {expected_hash} for {item[file_name]}\n{sample}"
    )


def candidate_urls(item: dict) -> list[str]:
    urls = [item["download_link"]]
    image_url = resolve_unsplash_image_url(item["download_link"])
    if image_url is None:
        return urls

    parsed_download = urllib.parse.urlparse(item["download_link"])
    download_params = urllib.parse.parse_qs(parsed_download.query)
    ixid = first(download_params.get("ixid"))

    parsed_image = urllib.parse.urlparse(image_url)
    base = urllib.parse.urlunparse((parsed_image.scheme, parsed_image.netloc, parsed_image.path, "", "", ""))
    current_params = dict(urllib.parse.parse_qsl(parsed_image.query))

    variants: list[dict[str, str]] = []
    if current_params:
        variants.append(current_params)
        no_crop = {k: v for k, v in current_params.items() if k != "crop"}
        variants.append(no_crop)

    if ixid:
        variants.extend(
            [
                {
                    "ixid": ixid,
                    "ixlib": "rb-4.1.0",
                    "q": "85",
                    "fm": "jpg",
                    "crop": "entropy",
                    "cs": "srgb",
                    "dl": item["file_name"],
                },
                {
                    "ixid": ixid,
                    "ixlib": "rb-4.1.0",
                    "q": "85",
                    "crop": "entropy",
                    "cs": "srgb",
                    "dl": item["file_name"],
                },
                {
                    "ixid": ixid,
                    "ixlib": "rb-4.1.0",
                    "q": "90",
                    "fm": "jpg",
                    "crop": "entropy",
                    "cs": "srgb",
                },
            ]
        )

    variants.extend(
        [
            {"q": "85", "fm": "jpg", "dl": item["file_name"]},
            {"fm": "jpg", "dl": item["file_name"]},
            {},
        ]
    )

    seen = set(urls)
    for params in variants:
        url = base
        if params:
            url = f"{base}?{urllib.parse.urlencode(params)}"
        if url not in seen:
            urls.append(url)
            seen.add(url)
    return urls


def resolve_unsplash_image_url(download_link: str) -> str | None:
    request = urllib.request.Request(download_link, method="HEAD", headers=headers())
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if "images.unsplash.com" in response.url:
                return response.url
    except Exception:
        return None
    return None



def fetch_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers=headers())
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def headers() -> dict[str, str]:
    return {"User-Agent": "Mozilla/5.0"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def first(values: list[str] | None) -> str | None:
    if not values:
        return None
    return values[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-file", type=pathlib.Path)
    parser.add_argument("--cache-dir", type=pathlib.Path)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
