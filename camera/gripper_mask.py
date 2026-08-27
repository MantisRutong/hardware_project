#!/usr/bin/env python3
"""Blank out the wrist-mounted gripper before stereo frames reach ORB-SLAM3.

## Why

The camera is wrist-mounted, so the gripper is rigidly attached to it and
sits in frame permanently -- measured at ~8.4% of the image (see
assets/gripper_mask.json for how that was derived). Features on it are
camera-fixed: they don't just fail to help, they are actively wrong. A
feature that never moves in the image is, geometrically, evidence that the
camera never moved, so gripper features systematically bias the translation
estimate DOWNWARD.

That matters here beyond general accuracy. The reset that broke live
pick-and-place testing is LocalMapping.cc's IMU-init gate, which needs
accumulated keyframe-to-keyframe translation >= 2cm within ~10s. Precise
manipulation is already close to that bar; anything suppressing the
translation estimate pushes it further under. (Plausible contributing
factor, not a measured one -- it has not been isolated experimentally.)

It also poisons a saved atlas. Gripper features stereo-triangulate to a
real distance from the camera, so ORB-SLAM3 happily creates map points for
them -- at whatever world position the camera occupied at the time. Those
points are then stored in atlas.osa, and on a later run the gripper (in the
same image location, as always) can match against them and drag
relocalization to the wrong pose.

This mirrors what the reference UMI pipeline does -- scripts_slam_pipeline/
03_batch_slam.py builds a slam_mask.png over its mirrors and fingers and
passes it to ORB-SLAM3 as --mask_img.

## Why mask in Python rather than in ORB-SLAM3

Upstream ORB-SLAM3 cannot do it: ORBextractor's operator() takes a _mask
argument and ignores it outright ("Mask is ignored in the current
implementation." -- include/ORBextractor.h), and System::TrackStereo has no
mask parameter at all. UMI's --mask_img comes from their own fork. Since
this project already owns the images before handing them to the C API,
flattening the region here needs no C++ change and no rebuild.

The cost of doing it this way is that the masked region's BORDER is itself
image structure, and a hard-edged polygon would just trade gripper corners
for boundary corners -- equally camera-fixed, equally wrong. So the fill is
feathered: alpha ramps smoothly over ~6px, and the fill value tracks the
frame's own brightness, leaving a gentle gradient rather than a step edge
for FAST to bite on. Not free of artifacts, just far weaker ones. Whether
what remains matters has NOT been validated on hardware.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_SPEC_PATH = Path(__file__).resolve().parent / "assets" / "gripper_mask.json"

# Feather width (pixels, Gaussian sigma) for the mask boundary -- see the
# module docstring's last paragraph. Wider = weaker boundary gradient, but
# also eats more real scene texture just outside the gripper. 6px is a
# starting point chosen to span a few ORB patch widths, not a measured
# optimum; revisit alongside a live tracked/total comparison.
FEATHER_SIGMA = 6.0


class GripperMask:
    """Feathered gripper cutout for one IR stream, cached per resolution.

    Built once and reused: the gripper is bolted to the camera, so its
    image footprint is fixed and there is nothing to re-detect per frame.
    """

    def __init__(self, stream: str, spec_path: Path = DEFAULT_SPEC_PATH) -> None:
        spec = json.loads(Path(spec_path).read_text())
        if stream not in spec:
            raise KeyError(f"{spec_path} has no polygons for stream {stream!r}")
        self.stream = stream
        self.spec_path = Path(spec_path)
        self.polygons = [np.asarray(p, dtype=np.int32) for p in spec[stream]]
        self.reference_resolution = tuple(spec["resolution"])  # (w, h)
        self._alpha_cache: dict[tuple[int, int], np.ndarray] = {}
        self._box_cache: dict[tuple[int, int], tuple[int, int, int, int] | None] = {}

    def _alpha(self, shape: tuple[int, int]) -> np.ndarray:
        """Feathered coverage map for (h, w): 1.0 fully masked, 0.0 untouched."""
        cached = self._alpha_cache.get(shape)
        if cached is not None:
            return cached
        h, w = shape
        ref_w, ref_h = self.reference_resolution
        # Scale rather than refuse, so a resolution fallback (see IR_PROFILES
        # in camera_collecting.py) still gets masked. The gripper's footprint
        # scales with the image because it's the same lens and same rigid
        # mount -- only the sampling changes.
        sx, sy = w / ref_w, h / ref_h
        hard = np.zeros(shape, dtype=np.uint8)
        for poly in self.polygons:
            scaled = np.round(poly * np.array([sx, sy])).astype(np.int32)
            cv2.fillPoly(hard, [scaled], 255)
        alpha = cv2.GaussianBlur(hard.astype(np.float32) / 255.0, (0, 0), FEATHER_SIGMA * min(sx, sy))
        self._alpha_cache[shape] = alpha
        # Bounding box of everything the feather actually touches. The mask
        # covers ~7% of the frame in one bottom-attached blob, so blending
        # only inside this box instead of over the whole image is a real
        # saving on a thread whose call budget is already watched closely
        # (see OrbSlamWorker's min_interval comment in synced_capture.py).
        ys, xs = np.where(alpha > 1e-3)
        self._box_cache[shape] = (
            (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
            if len(ys) else None
        )
        return alpha

    def apply(self, image: Any) -> np.ndarray:
        """Return a copy of `image` with the gripper flattened away.

        Never modifies the input: the same array is also on its way to disk
        as the episode's ir_left/ir_right PNGs, which must stay pristine --
        they're the raw record, and any later offline pass (re-running SLAM,
        re-deriving this very mask) needs to see what the camera actually
        saw, not what tracking was fed.
        """
        img = np.asarray(image)
        shape = img.shape[:2]
        alpha = self._alpha(shape)
        box = self._box_cache[shape]
        out = img.copy()
        if box is None:
            return out
        y0, y1, x0, x1 = box
        # Fill with the frame's own brightness OUTSIDE the mask, so the patch
        # sits at roughly the scene's level rather than introducing a bright
        # or dark slab whose feathered edge is a strong gradient in its own
        # right. Sampled from the untouched part of the same rows the mask
        # occupies, not the whole frame: with a downward-looking wrist camera
        # the top of the image is often a different surface at a different
        # distance and brightness entirely.
        band = img[y0:y1]
        band_alpha = alpha[y0:y1]
        w = 1.0 - band_alpha
        denom = float(w.sum())
        fill = float((band * w).sum() / denom) if denom > 1e-6 else float(img.mean())
        sub, sub_alpha = img[y0:y1, x0:x1].astype(np.float32), alpha[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = (sub * (1.0 - sub_alpha) + fill * sub_alpha).astype(img.dtype)
        return out

    def coverage(self, shape: tuple[int, int]) -> float:
        """Fraction of the frame fully masked -- for logging/sanity checks."""
        return float((self._alpha(shape) > 0.99).mean())


def load_stereo_masks(spec_path: Path = DEFAULT_SPEC_PATH) -> tuple[GripperMask, GripperMask]:
    """(left, right) masks. Separate polygons because the two IR cameras see
    the gripper from different viewpoints -- their outlines differ by roughly
    the stereo disparity."""
    return GripperMask("ir_left", spec_path), GripperMask("ir_right", spec_path)


def main() -> int:
    """Preview the mask over real frames, to eyeball it after a remount."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("scan_dir", type=Path, help="A recording dir containing ir_left/ and ir_right/.")
    parser.add_argument("--stream", default="ir_left", choices=("ir_left", "ir_right"))
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--out", type=Path, default=Path("gripper_mask_preview.png"))
    parser.add_argument("--tiles", type=int, default=6)
    args = parser.parse_args()

    files = sorted((args.scan_dir / args.stream).glob("*.png"))
    if not files:
        raise SystemExit(f"no frames in {args.scan_dir / args.stream}")
    mask = GripperMask(args.stream, args.spec)

    tiles = []
    for i in np.linspace(0, len(files) - 1, args.tiles).astype(int):
        raw = cv2.imread(str(files[i]), cv2.IMREAD_GRAYSCALE)
        pair = np.hstack([raw, mask.apply(raw)])
        tiles.append(cv2.resize(pair, (pair.shape[1] // 2, pair.shape[0] // 2)))
    grid = np.vstack(tiles)
    cv2.imwrite(str(args.out), grid)
    h, w = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE).shape
    print(f"{args.stream}: masks {100 * mask.coverage((h, w)):.1f}% of a {w}x{h} frame "
          f"({len(mask.polygons)} polygon(s), feather sigma {FEATHER_SIGMA}px)")
    print(f"wrote {args.out} -- each row is raw | masked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
