"""Fetch + extract raw MIND / EB-NeRD files. Idempotent: skips work whose output
already exists on disk unless force=True.
"""

import subprocess
import zipfile
from pathlib import Path

import requests
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "datasets.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _download_file(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        print(f"  [skip] {dest} already exists")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return dest


def _unzip(zip_path: Path, extract_dir: Path, force: bool = False) -> Path:
    # A per-zip marker, not just "does extract_dir have anything in it" -- EB-NeRD's
    # large-scale download extracts 3 different zips into the *same* extract_dir, so a
    # directory-non-emptiness check can't tell "this zip is done" from "some other zip
    # already put things here" and silently skips real work.
    marker = extract_dir / f".extracted_{zip_path.stem}"
    if marker.exists() and not force:
        print(f"  [skip] {zip_path.name} already extracted into {extract_dir}")
        return extract_dir
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    marker.touch()
    return extract_dir


def download_ebnerd(scale: str, raw_dir: Path, config: dict, force: bool = False) -> Path:
    cfg = config["ebnerd"][scale]
    dataset_raw = raw_dir / "ebnerd" / scale
    zip_path = raw_dir / "ebnerd" / f"{scale}.zip"

    _download_file(cfg["behaviors_zip_url"], zip_path, force=force)
    _unzip(zip_path, dataset_raw, force=force)

    if "articles_zip_url" in cfg:
        art_zip = raw_dir / "ebnerd" / f"{scale}_articles.zip"
        _download_file(cfg["articles_zip_url"], art_zip, force=force)
        _unzip(art_zip, dataset_raw, force=force)

    if "testset_zip_url" in cfg:
        test_zip = raw_dir / "ebnerd" / f"{scale}_testset.zip"
        _download_file(cfg["testset_zip_url"], test_zip, force=force)
        _unzip(test_zip, dataset_raw, force=force)

    return dataset_raw


def _hf_download_zip(repo: str, zip_name: str, local_dir: Path, force: bool = False) -> Path:
    zip_path = local_dir / zip_name
    if zip_path.exists() and not force:
        print(f"  [skip] {zip_path} already exists")
        return zip_path
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"  hf download {repo} {zip_name} -> {local_dir}")
    subprocess.run(
        ["hf", "download", repo, zip_name, "--repo-type", "dataset", "--local-dir", str(local_dir)],
        check=True,
    )
    return zip_path


def download_mind(scale: str, raw_dir: Path, config: dict, force: bool = False) -> Path:
    cfg = config["mind"]
    repo = cfg["hf_repo"]
    dataset_raw = raw_dir / "mind"
    scale_cfg = cfg[scale]

    for key, zip_name in scale_cfg.items():
        split = key.replace("_zip", "")  # train/dev/test
        zip_path = _hf_download_zip(repo, zip_name, dataset_raw, force=force)
        extract_dir = dataset_raw / zip_name.replace(".zip", "")
        _unzip(zip_path, extract_dir, force=force)

    return dataset_raw


def download(dataset: str, scale: str, raw_dir: Path, config: dict, force: bool = False) -> Path:
    if dataset == "ebnerd":
        return download_ebnerd(scale, raw_dir, config, force=force)
    elif dataset == "mind":
        return download_mind(scale, raw_dir, config, force=force)
    raise ValueError(f"unknown dataset {dataset!r}")
