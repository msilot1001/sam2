"""Download, verify and selectively extract Virtual KITTI 2.0.3.

Stdlib only + curl, so it runs before the venv exists.

    python scripts/setup_vkitti.py                       # Scene01 clone Camera_0
    python scripts/setup_vkitti.py --scenes Scene01 Scene06 --cameras 0
    python scripts/setup_vkitti.py --skip-download       # re-extract only

Safe to re-run: downloads resume from *.part, verified archives are skipped,
extraction skips files that already exist with the same size.
"""
import argparse
import hashlib
import re
import subprocess
import sys
import tarfile
from pathlib import Path

BASE_URL = "https://download.europe.naverlabs.com/virtual_kitti_2.0.3"
# Official sizes (HTTP Content-Length) and MD5s (vkitti_2.0.3_md5_checksums.txt), checked 2026-10-12.
ARCHIVES = {  # small first, so text/mask files are usable while RGB downloads
    "instance": ("vkitti_2.0.3_instanceSegmentation.tar", 172298240, "c79684c6344e50663587b8da848535e7"),
    "textgt": ("vkitti_2.0.3_textgt.tar.gz", 24451078, "855f7f37746e5508f094e522c9cf41f4"),
    "rgb": ("vkitti_2.0.3_rgb.tar", 7532472320, "1e00a143a397c2c53aa9720a868fe34a"),
}
SCENE_RE = re.compile(r"(Scene\d{2}/.*)$")


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, size: int, md5: str, archive_dir: Path) -> Path:
    dst = archive_dir / name
    if dst.exists():
        if dst.stat().st_size == size:
            print(f"[ok] {name} already present")
            return dst
        sys.exit(f"[error] {dst} has size {dst.stat().st_size}, expected {size}. Delete it and re-run.")
    part = dst.with_name(name + ".part")
    print(f"[download] {name} ({size / 1e9:.2f} GB) -> {part}")
    # curl -C - resumes from the existing .part; retries cover transient network failures.
    subprocess.run(
        ["curl", "-L", "--fail", "-C", "-", "--retry", "10", "--retry-all-errors", "--retry-delay", "5",
         "--connect-timeout", "30", "--speed-time", "60", "--speed-limit", "1024",
         "--progress-bar", "-o", str(part), f"{BASE_URL}/{name}"],
        check=True,
    )
    got = part.stat().st_size
    if got != size:
        sys.exit(f"[error] {part}: size {got} != expected {size}. Re-run to resume.")
    print(f"[md5] verifying {name} ...")
    digest = md5sum(part)
    if digest != md5:
        sys.exit(f"[error] {part}: md5 {digest} != official {md5}. Delete the .part file and re-run.")
    part.rename(dst)
    print(f"[ok] {name} size and md5 verified")
    return dst


def wanted(rel: str, kind: str, scenes, variations, cameras) -> bool:
    parts = rel.split("/")
    if len(parts) < 3 or parts[0] not in scenes or parts[1] not in variations:
        return False
    if kind == "textgt":
        return True  # small per-scene text files, keep all cameras
    return any(f"Camera_{c}" in parts for c in cameras)


def extract(archive: Path, kind: str, out: Path, scenes, variations, cameras) -> int:
    """Extract matching members, normalising whatever prefix precedes SceneXX/."""
    n = 0
    with tarfile.open(archive, "r:*") as tar:  # streaming read; no full index needed
        for m in tar:
            match = SCENE_RE.search(m.name)
            if not m.isfile() or not match or not wanted(match.group(1), kind, scenes, variations, cameras):
                continue
            rel = match.group(1)
            if ".." in Path(rel).parts:
                continue
            dst = out / rel
            if dst.exists() and dst.stat().st_size == m.size:
                n += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".tmp")
            with tar.extractfile(m) as src, tmp.open("wb") as f:
                f.write(src.read())
            tmp.rename(dst)
            n += 1
    print(f"[extract] {archive.name}: {n} files for {scenes}/{variations}/cameras {cameras}")
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("data/vkitti2"))
    p.add_argument("--scenes", nargs="+", default=["Scene01"])
    p.add_argument("--variations", nargs="+", default=["clone"])
    p.add_argument("--cameras", nargs="+", type=int, default=[0])
    p.add_argument("--skip-download", action="store_true")
    a = p.parse_args()

    archive_dir = a.root / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for kind, (name, size, md5) in ARCHIVES.items():
        path = archive_dir / name
        if not a.skip_download:
            path = download(name, size, md5, archive_dir)
        elif not path.exists():
            sys.exit(f"[error] {path} missing; run without --skip-download")
        if extract(path, kind, a.root, a.scenes, a.variations, a.cameras) == 0:
            sys.exit(f"[error] no members matched in {name}; check --scenes/--variations/--cameras")


if __name__ == "__main__":
    main()
