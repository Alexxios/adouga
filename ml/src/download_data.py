"""Download training data ZIP archives from Yandex Disk.

Reads the OAuth token from the ``YADISK_TOKEN`` environment variable
(or a ``.env`` file).

Usage::

    python -m src.download_data
    python -m src.download_data --output data/zips --remote /adouga/ml_sessions
"""

import argparse
import os
import sys
from pathlib import Path

import yadisk
from dotenv import load_dotenv

_REMOTE_BASE = "/adouga/ml_sessions"
_DEFAULT_OUTPUT = "data/zips"
_TIMEOUT = 30  # seconds per request


def _resolve_token() -> str:
    load_dotenv()
    token = os.environ.get("YADISK_TOKEN")
    if not token:
        print("YADISK_TOKEN is not set. Add it to your environment or .env file.")
        sys.exit(1)
    return token


def download_all(remote_base: str, output_dir: Path, exclude: set[str] | None = None) -> None:
    token = _resolve_token()
    output_dir.mkdir(parents=True, exist_ok=True)
    exclude = exclude or set()

    with yadisk.Client(token=token) as client:
        if not client.check_token():
            print("Invalid YADISK_TOKEN — authentication failed.")
            sys.exit(1)

        print(f"Connected to Yandex Disk")
        print(f"Scanning {remote_base}/ ...")
        if exclude:
            print(f"Excluding directories: {sorted(exclude)}")

        zip_files = list(_find_zips(client, remote_base, exclude))

        if not zip_files:
            print("No ZIP files found.")
            return

        print(f"Found {len(zip_files)} ZIP archive(s)\n")

        downloaded = 0
        skipped = 0
        failed = 0

        for i, remote_path in enumerate(zip_files, 1):
            filename = remote_path.rsplit("/", 1)[-1]
            local_path = output_dir / filename
            tmp_path = output_dir / (filename + ".part")

            if local_path.exists():
                print(f"  [{i}/{len(zip_files)}] {filename} — already exists, skipping")
                skipped += 1
                continue

            print(f"  [{i}/{len(zip_files)}] Downloading {filename} ...", end=" ", flush=True)
            try:
                # Download to temp file first; rename on success
                client.download(remote_path, str(tmp_path), timeout=_TIMEOUT)
                tmp_path.rename(local_path)
                size_mb = local_path.stat().st_size / (1024 * 1024)
                print(f"{size_mb:.1f} MB")
                downloaded += 1
            except Exception as exc:
                print(f"FAILED ({exc})")
                tmp_path.unlink(missing_ok=True)
                failed += 1

        print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {failed} failed")


def _find_zips(client: yadisk.Client, path: str, exclude: set[str]):
    """Recursively find all .zip files under a remote directory."""
    try:
        for item in client.listdir(path):
            if item.type == "dir":
                if item.name in exclude:
                    print(f"  Skipping excluded directory: {item.path}")
                    continue
                yield from _find_zips(client, item.path, exclude)
            elif item.name.endswith(".zip"):
                yield item.path
    except yadisk.exceptions.PathNotFoundError:
        print(f"Remote path not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download training data from Yandex Disk")
    parser.add_argument("--output", "-o", default=_DEFAULT_OUTPUT, help="Local output directory")
    parser.add_argument("--remote", "-r", default=_REMOTE_BASE, help="Remote base directory on Yandex Disk")
    parser.add_argument(
        "--exclude",
        "-x",
        action="append",
        default=[],
        help="Directory name to skip during traversal (can be passed multiple times)",
    )
    args = parser.parse_args()

    download_all(args.remote, Path(args.output), set(args.exclude))


if __name__ == "__main__":
    main()
