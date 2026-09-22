"""Download everything needed to run the f-CBM experiments into `data/`.

Step 1 - raw images, fetched from their official sources (never re-hosted by us):
    * CUB-200-2011 : Caltech DATA            (1.1 GB, md5-checked)
    * N24News      : authors' Google Drive   (6.5 GB)
Step 2 - our concept annotations (and optionally the trained checkpoints),
    fetched from the Hugging Face Hub. AG News and DBpedia only need this step:
    their annotation files already contain the text and the labels.

Usage (from the f-CBM/ folder):
    python scripts/download_data.py                          # everything
    python scripts/download_data.py --datasets agnews dbpedia
    python scripts/download_data.py --with-checkpoints       # + trained models (~9 GB)
    python scripts/download_data.py --hf-repo <user>/f-cbm-data   # if the repo id is not set below

The script is idempotent: what is already on disk is skipped, and an
interrupted download resumes where it stopped.
"""
import argparse
import hashlib
import os
import sys
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- step 1: official sources
CUB = dict(
    url="https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
    archive="CUB_200_2011.tgz",
    size=1_150_585_339,
    md5="97eceeb196236b17998738112f37df78",  # published by Caltech DATA
    n_images=11_788,
    page="https://data.caltech.edu/records/65de6-vp158",
)
N24NEWS = dict(
    url="https://drive.usercontent.google.com/download?id=1OS1fXwZ1Vsj70lEQajccyssxQRYp5X9D&export=download&confirm=t",
    archive="N24News.zip",
    size=7_000_660_656,
    n_images=61_236,
    page="https://github.com/billywzh717/N24News",
)

# ---------------------------------------------------------------- step 2: our files
# Hub repo holding the content of `data/` minus the raw images. The revision is
# pinned so that everyone gets exactly the files used in the paper.
HF_REPO_ID = os.environ.get("FCBM_HF_REPO", "")  # TODO: set to "<user>/f-cbm-data" once uploaded
HF_REVISION = "v1.0"

DATASET_DIRS = {"cub": "CUB_200_2011", "n24news": "N24News", "agnews": "agnews", "dbpedia": "dbpedia"}
RAW_PATTERNS = ["datasets/CUB_200_2011/images/**", "datasets/N24News/imgs/**", "datasets/N24News/news/**"]
CHECKPOINT_PATTERNS = ["*.pt", "*.pth"]


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def count_jpg(folder):
    return sum(1 for _, _, files in os.walk(folder) for f in files if f.lower().endswith(".jpg"))


def download(src, dest):
    """Stream src['url'] to dest, resuming from dest.part if a previous run was interrupted."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    pos = part.stat().st_size if part.exists() else 0

    if pos < src["size"]:
        headers = {"Range": f"bytes={pos}-"} if pos else {}
        with requests.get(src["url"], headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            if "text/html" in r.headers.get("Content-Type", ""):
                # Google Drive answers with an HTML page when its daily quota is exceeded.
                sys.exit(f"\n{dest.name}: the server did not return the file (download quota exceeded?).\n"
                         f"Download it by hand from {src['page']}\nsave it as {dest}\nand run this script again.")
            if pos and r.status_code != 206:  # server ignored the Range header: start over
                pos = 0
            with open(part, "ab" if pos else "wb") as f, tqdm(
                    total=src["size"], initial=pos, unit="B", unit_scale=True, desc=dest.name) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))

    if part.stat().st_size != src["size"]:
        sys.exit(f"{dest.name}: got {part.stat().st_size} bytes, expected {src['size']}. Run the script again to resume.")
    if "md5" in src:
        print(f"{dest.name}: checking md5 ...")
        if md5sum(part) != src["md5"]:
            part.unlink()
            sys.exit(f"{dest.name}: md5 mismatch, the corrupted file was removed. Run the script again.")
    part.rename(dest)


def get_cub(data_dir, keep_archive):
    images = data_dir / "datasets" / "CUB_200_2011" / "images"
    if images.exists() and count_jpg(images) >= CUB["n_images"]:
        print("CUB-200-2011 images: already there.")
        return
    archive = data_dir / "_downloads" / CUB["archive"]
    download(CUB, archive)
    with tarfile.open(archive) as tar:
        members = [m for m in tar if m.name.startswith("CUB_200_2011/images/")]
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        for m in tqdm(members, desc="extracting CUB images", unit="file"):
            tar.extract(m, data_dir / "datasets", **kwargs)
    assert count_jpg(images) >= CUB["n_images"], "CUB extraction is incomplete"
    if not keep_archive:
        archive.unlink()


def get_n24news(data_dir, keep_archive):
    target = data_dir / "datasets" / "N24News"
    if (target / "imgs").exists() and count_jpg(target / "imgs") >= N24NEWS["n_images"] \
            and (target / "news" / "nytimes_dataset.json").exists():
        print("N24News images: already there.")
        return
    archive = data_dir / "_downloads" / N24NEWS["archive"]
    download(N24NEWS, archive)
    with zipfile.ZipFile(archive) as z:  # the archive holds imgs/ and news/, as the code expects
        for m in tqdm(z.infolist(), desc="extracting N24News", unit="file"):
            z.extract(m, target)
    assert count_jpg(target / "imgs") >= N24NEWS["n_images"], "N24News extraction is incomplete"
    if not keep_archive:
        archive.unlink()


def get_annotations(data_dir, datasets, with_checkpoints, repo_id):
    if not repo_id:
        sys.exit("No Hugging Face repo id: the concept annotations are not published yet.\n"
                 "Once they are, pass --hf-repo <user>/f-cbm-data (or set FCBM_HF_REPO); see README.md, section Data.")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("Missing dependency: pip install huggingface_hub")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=HF_REVISION,
        local_dir=data_dir,
        allow_patterns=[f"datasets/{DATASET_DIRS[d]}/**" for d in datasets],
        ignore_patterns=RAW_PATTERNS + ([] if with_checkpoints else CHECKPOINT_PATTERNS),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_DIRS), default=list(DATASET_DIRS))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="must match PATH in src/path_info.py (default: f-CBM/data)")
    parser.add_argument("--with-checkpoints", action="store_true",
                        help="also fetch the trained models, to skip the training of the backbones")
    parser.add_argument("--skip-images", action="store_true", help="skip step 1 (official sources)")
    parser.add_argument("--skip-annotations", action="store_true", help="skip step 2 (Hugging Face Hub)")
    parser.add_argument("--keep-archives", action="store_true", help="keep the .tgz/.zip after extraction")
    parser.add_argument("--hf-repo", default=HF_REPO_ID,
                        help="Hugging Face dataset repo holding the annotations (default: $FCBM_HF_REPO)")
    args = parser.parse_args()

    if not args.skip_images:
        if "cub" in args.datasets:
            get_cub(args.data_dir, args.keep_archives)
        if "n24news" in args.datasets:
            get_n24news(args.data_dir, args.keep_archives)
    if not args.skip_annotations:
        get_annotations(args.data_dir, args.datasets, args.with_checkpoints, args.hf_repo)
    print(f"Done. Data is in {args.data_dir}")


if __name__ == "__main__":
    main()
