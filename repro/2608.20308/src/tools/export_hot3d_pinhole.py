"""Write one HOT3D clip tar as an upright pinhole mp4.

Same undistort and clockwise-90° rotation as ``converters/hot3d.py``. The
released checkpoints do not take the native Aria fisheye. This script does
not encode a ``.pt``; pass ``--video-dir`` to the converter when the latent
clip is needed too.

    cd repro/2608.20308/src
    python tools/export_hot3d_pinhole.py \\
        ../data/hot3d/raw/train_aria/clip-001849.tar \\
        ../data/hot3d/pinhole/clip-001849.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))

from ace_repro.data.converters.hot3d import (  # noqa: E402
    STREAM,
    undistort_map,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tar", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--focal-scale", type=float, default=1.0)
    args = ap.parse_args()

    tar = tarfile.open(args.tar)
    cam0 = json.load(tar.extractfile("000000.cameras.json"))[STREAM]
    params = cam0["calibration"]["projection_params"]
    focal, cx, cy = params[:3]
    coeff = np.asarray(params[3:], dtype=np.float32)
    height = int(cam0["calibration"]["image_height"])
    width = int(cam0["calibration"]["image_width"])
    map_x, map_y, _ = undistort_map(height, width, focal, (cx, cy), coeff, args.focal_scale)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # rot90 clockwise swaps H and W.
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (height, width))
    if not writer.isOpened():
        raise SystemExit(f"failed to open {args.out}")
    for t in range(args.frames):
        blob = tar.extractfile(f"{t:06d}.image_{STREAM}.jpg").read()
        bgr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"frame {t} failed to decode")
        pinhole = cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        writer.write(cv2.rotate(pinhole, cv2.ROTATE_90_CLOCKWISE))
    writer.release()
    tar.close()
    print(f"wrote {args.out} {args.frames} frames {height}x{width} upright pinhole", flush=True)


if __name__ == "__main__":
    main()
