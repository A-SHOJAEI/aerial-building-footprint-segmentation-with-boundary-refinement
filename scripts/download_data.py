"""Download Massachusetts Buildings dataset from University of Toronto."""

import re
from pathlib import Path

import requests
from tqdm import tqdm


BASE_URL = "https://www.cs.toronto.edu/~vmnih/data/mass_buildings"


def download_file(url: str, dest: Path):
    """Download file with progress bar."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
    ) as pbar:
        for chunk in response.iter_content(8192):
            f.write(chunk)
            pbar.update(len(chunk))


def get_file_list(url: str) -> list[str]:
    """Parse index.html to get list of filenames."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    # Extract href links - they may be full URLs or just filenames
    hrefs = re.findall(r'href="([^"]+\.tiff?)"', resp.text)
    # Extract just the filename from each href (handles full URLs)
    filenames = []
    for href in hrefs:
        # Get the last path component (filename)
        name = href.rstrip("/").rsplit("/", 1)[-1]
        if name:
            filenames.append(name)
    return filenames


def download_split(data_dir: Path, split: str):
    """Download satellite images and masks for a split."""
    sat_dir = data_dir / split / "sat"
    map_dir = data_dir / split / "map"
    sat_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    # Get file lists from index pages
    sat_url = f"{BASE_URL}/{split}/sat/index.html"
    map_url = f"{BASE_URL}/{split}/map/index.html"

    sat_files = get_file_list(sat_url)
    map_files = get_file_list(map_url)

    print(f"  {split}: {len(sat_files)} sat, {len(map_files)} map files")

    # Download satellite images
    existing_sat = set(f.name for f in sat_dir.iterdir() if f.suffix in ('.tiff', '.tif'))
    to_download = [f for f in sat_files if f not in existing_sat]
    if to_download:
        print(f"  Downloading {len(to_download)} satellite images...")
        for fname in tqdm(to_download, desc=f"{split}/sat"):
            url = f"{BASE_URL}/{split}/sat/{fname}"
            download_file(url, sat_dir / fname)
    else:
        print(f"  {split}/sat: already complete")

    # Download masks
    existing_map = set(f.name for f in map_dir.iterdir() if f.suffix in ('.tiff', '.tif'))
    to_download = [f for f in map_files if f not in existing_map]
    if to_download:
        print(f"  Downloading {len(to_download)} mask images...")
        for fname in tqdm(to_download, desc=f"{split}/map"):
            url = f"{BASE_URL}/{split}/map/{fname}"
            download_file(url, map_dir / fname)
    else:
        print(f"  {split}/map: already complete")


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "mass_buildings"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading Massachusetts Buildings Dataset")
    print("Source: www.cs.toronto.edu/~vmnih/data/mass_buildings/")
    print("=" * 60)

    for split in ["train", "valid", "test"]:
        print(f"\n[{split.upper()}]")
        download_split(data_dir, split)

    # Summary
    print("\n" + "=" * 60)
    for split in ["train", "valid", "test"]:
        sat_dir = data_dir / split / "sat"
        map_dir = data_dir / split / "map"
        n_sat = len(list(sat_dir.glob("*.*"))) if sat_dir.exists() else 0
        n_map = len(list(map_dir.glob("*.*"))) if map_dir.exists() else 0
        print(f"  {split}: {n_sat} images, {n_map} masks")
    print("Dataset ready!")


if __name__ == "__main__":
    main()
