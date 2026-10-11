"""Ad-hoc RGB + track-mask overlays.

    uv run python scripts/visualize_vkitti.py --track 4 --frames 0 50 100
"""
import argparse
from pathlib import Path

from vkitti.dataset import VKittiDataset
from vkitti.visualization import save_overlay

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/vkitti2")
ap.add_argument("--scene", default="Scene01")
ap.add_argument("--variation", default="clone")
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--track", type=int, required=True)
ap.add_argument("--frames", type=int, nargs="+", required=True)
ap.add_argument("--out", type=Path, default=Path("outputs/milestone0/overlays"))
a = ap.parse_args()

ds = VKittiDataset(a.root, a.scene, a.variation, a.camera)
a.out.mkdir(parents=True, exist_ok=True)
for f in a.frames:
    p = a.out / f"{ds.tag.replace('/', '_')}_track{a.track:03d}_frame{f:05d}.png"
    save_overlay(ds.get_frame(f, a.track), p)
    print(p)
