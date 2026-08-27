#!/usr/bin/env python3
"""Build a persistent ORB-SLAM3 map (an "atlas" .osa file) of a workspace,
via one deliberate, expansive live scan -- the first half of a two-stage
tracking approach: build the map ONCE here, then have every individual
demonstration recording LOCALIZE against that already-built map instead of
trying to cold-start a fresh map from the demo's own motion.

## Why this exists

Live testing on this project repeatedly showed ORB-SLAM3 Stereo-Inertial
failing catastrophically (near-constant resets, ~33% frames tracked, every
"tracked" segment only ~3 frames long) specifically during careful,
precise small-object pick-and-place -- even after independently ruling out
low texture and motion blur as the cause (both fixed, no change in the
failure pattern). Root cause, traced into ORB-SLAM3's own source
(LocalMapping.cc): IMU initialization requires accumulated keyframe-to-
keyframe translation >= 2cm within a ~10s window, or it resets
("Not enough motion for initializing...") -- and precise manipulation
naturally involves small net displacements, so a fresh map cold-started
from inside a single short, careful demo may simply never clear that bar.

The official UMI pipeline (~/projects/diffusion_policy/third_party/
universal_manipulation_interface/scripts_slam_pipeline/03_batch_slam.py)
never hits this: it builds a persistent map ONCE, offline, from a
separate deliberate mapping pass (expansive motion, well clear of the
init threshold), then tracks every actual demo by LOADING that map and
localizing against it -- never cold-starting a map from a demo's own
(often too-careful) motion. This script is the live equivalent of that
mapping pass for this project's own D435i/ORB-SLAM3 setup.

The localize half is NOT a separate script -- it's `synced_capture.py
--orbslam-map-dir <this script's --output-dir>`, which loads the atlas
built here instead of cold-starting a fresh map per episode (see that
flag's help, `construct_orb_tracker()` in that file, and
ORB_SLAM3/config/RealSense_D435i_ours_relocalize.yaml's comment for the
exact mechanism -- briefly: the saved atlas serializes `mbIMU_BA2`, and
the reset that broke live testing is gated on `!GetIniertialBA2()`, so a
loaded already-initialized map never re-triggers it).

## How saving actually works

ORB_SLAM3::System::Shutdown() automatically calls SaveAtlas() if the
settings YAML has a non-empty System.SaveAtlasToFile value (see
ORB_SLAM3/src/System.cc) -- no C API changes needed, just pointing at
ORB_SLAM3/config/RealSense_D435i_ours_mapping.yaml (identical calibration
to RealSense_D435i_ours.yaml, plus that one key). ORB-SLAM3 hardcodes the
save path as "./" + name + ".osa" (not configurable) -- this script
os.chdir()s into --output-dir before starting tracking so the file lands
exactly where asked, regardless of where the script itself was invoked
from.

## Usage

    python3 build_atlas_map.py --output-dir maps/kitchen_table

Move the camera through the WHOLE workspace a demo might later need to
localize against -- thoroughly, and with real (not overly careful)
motion, since map quality here directly determines whether later demos
can localize against it at all. Press Enter to stop; watch the printed
tracked/total ratio and (if --viewer is on, the default) ORB-SLAM3's own
Pangolin map viewer for a live sense of coverage before stopping.

Two things about the scene itself matter as much as the motion, because
both mapping and every later demo track on the IR streams:

- The IR emitter is now switched OFF whenever the IR streams feed
  ORB-SLAM3 (see camera_collecting.py's _set_emitter_for_depth) -- its
  projected dots are camera-fixed, so they were never valid landmarks.
  The scene therefore has to supply its OWN world-fixed texture.
- Adding texture deliberately (e.g. printed patterns stuck around the
  workspace) is the intended way to do that. They're used only as
  ordinary ORB features -- nothing detects them as markers, and no pose
  is derived from their geometry. Note they must be BLACK-ON-WHITE from
  a laser printer or similar: tracking happens in near-IR, where many
  colored dyes are effectively transparent.

Whatever texture the workspace has when this runs is baked into the
atlas as map points, so it must STAY PUT between this mapping pass and
the demos that localize against it. Rearranging it means re-running
this script.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "camera"))
sys.path.insert(0, str(REPO_ROOT / "output_script"))

import camera_collecting as cam  # noqa: E402
from orbslam_bridge import DEFAULT_LIB_PATH, DEFAULT_VOCAB_PATH, OrbSlamTracker  # noqa: E402
# Reuse the live-tracking worker from synced_capture.py -- its submission
# throttle (min_interval, default 10Hz) took several rounds of live crash-
# debugging to get right (see that class's own docstring); duplicating it
# here instead of importing would risk silently regressing the same bug.
from synced_capture import OrbSlamWorker  # noqa: E402

DEFAULT_MAPPING_SETTINGS = Path("/home/hakan/Desktop/umi/ORB_SLAM3/config/RealSense_D435i_ours_mapping.yaml")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="Where atlas.osa lands, plus a live sync/tracking log for this mapping session.")
    parser.add_argument("--settings", type=Path, default=DEFAULT_MAPPING_SETTINGS,
                         help=f"ORB-SLAM3 settings YAML with System.SaveAtlasToFile set. Default: {DEFAULT_MAPPING_SETTINGS}")
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--lib", type=Path, default=DEFAULT_LIB_PATH)
    parser.add_argument("--viewer", dest="viewer", action="store_true", default=True,
                         help="Show ORB-SLAM3's own Pangolin 3D map viewer (default: ON here -- unlike live "
                              "data collection, there's no servo/recording load competing for it, and seeing "
                              "map coverage live is the actual point of this script).")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--max-duration", type=float, default=None, help="Optional safety cap, seconds.")
    parser.add_argument("--min-interval", type=float, default=0.1,
                         help="OrbSlamWorker submission throttle, seconds (default 0.1s/10Hz -- see "
                              "synced_capture.py's OrbSlamWorker for why this exact value; don't lower it "
                              "without reading that class's docstring first).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Resolve everything that needs to survive the chdir below BEFORE it.
    settings_path = args.settings.resolve()
    vocab_path = args.vocab.resolve()
    lib_path = args.lib.resolve()
    output_dir = args.output_dir.resolve()
    if not settings_path.is_file():
        raise SystemExit(f"Settings file not found: {settings_path}")

    import pyrealsense2 as rs

    os.chdir(output_dir)  # see module docstring's "How saving actually works"

    capture = cam.RealSenseCapture(
        rs=rs,
        scan_dir=output_dir,
        max_duration_seconds=args.max_duration,
        requested_color=None,
        requested_depth=None,
        enable_imu=True,
        record_rgb=False,   # not needed to build the map -- only IR (via record_ir) feeds ORB-SLAM3
        record_depth=False,
        record_ir=True,     # required: gates both the IR stream itself and on_stereo_frame firing
        queue_size=256,
        imu_fps=200,
        show_preview=False,
        preview_fps=15.0,
    )
    capture.start_camera()

    tracker = OrbSlamTracker(settings_path=settings_path, vocab_path=vocab_path, lib_path=lib_path,
                              use_viewer=args.viewer)
    worker = None  # assigned below; declared here so the finally block can reference it either way

    last_gyro = [0.0, 0.0, 0.0]
    have_gyro = [False]
    last_print = [0.0]
    tracked_count = [0]
    total_count = [0]

    def _on_pose(ts: float, pose: tuple | None, map_epoch: int) -> None:
        total_count[0] += 1
        if pose is not None:
            tracked_count[0] += 1
        now = time.monotonic()
        if now - last_print[0] >= 1.0:
            last_print[0] = now
            if pose is not None:
                x, y, z = pose[0], pose[1], pose[2]
                print(f"\r[mapping] tracked {tracked_count[0]}/{total_count[0]}  "
                      f"pos=({x:.2f},{y:.2f},{z:.2f})  map_epoch={map_epoch}   ", end="", flush=True)
            else:
                print(f"\r[mapping] tracked {tracked_count[0]}/{total_count[0]}  (lost)  "
                      f"map_epoch={map_epoch}   ", end="", flush=True)

    worker = OrbSlamWorker(tracker, _on_pose, min_interval=args.min_interval)

    def _on_imu_sample(kind: str, row: dict[str, Any]) -> None:
        if kind == "gyro":
            last_gyro[0], last_gyro[1], last_gyro[2] = row["x"], row["y"], row["z"]
            have_gyro[0] = True
        elif kind == "accel" and have_gyro[0]:
            # Host clock, not the RealSense hardware timestamp -- see the
            # matching comment in synced_capture.py's _on_imu_sample.
            tracker.feed_imu(row["host_time_ns"] / 1e9, last_gyro[0], last_gyro[1], last_gyro[2],
                              row["x"], row["y"], row["z"])

    def _on_stereo_frame(frame_id: int, host_time_ns: int, left_gray: Any, right_gray: Any) -> None:
        ts = capture.frame_rows[-1]["host_time_ns"] / 1e9
        worker.submit(ts, left_gray, right_gray)

    capture.on_imu_sample = _on_imu_sample
    capture.on_stereo_frame = _on_stereo_frame

    print(f"Mapping session starting -- output: {output_dir}")
    print("Move the camera through the WHOLE workspace, thoroughly and with real (not overly careful) motion,")
    print("covering every area a demo might later need to localize against.")
    print("Press Enter to stop.\n")

    def _wait_for_stop(event: threading.Event = capture.stop_event) -> None:
        try:
            input()
        except EOFError:
            pass
        event.set()

    threading.Thread(target=_wait_for_stop, daemon=True).start()

    try:
        capture.capture()
    finally:
        worker.close()
        capture.stop_camera()
        print(f"\n\n{tracked_count[0]}/{total_count[0]} frames tracked during mapping.")
        print("Shutting down ORB-SLAM3 (this triggers the atlas save)...")
        tracker.shutdown()
        tracker.close()

    atlas_path = output_dir / "atlas.osa"
    if atlas_path.is_file():
        size_mb = atlas_path.stat().st_size / 1e6
        print(f"Wrote {atlas_path} ({size_mb:.1f} MB)")
        return 0
    print(f"WARNING: expected {atlas_path} but it wasn't created -- check the ORB-SLAM3 console output above "
          f"for errors, and confirm {settings_path} has System.SaveAtlasToFile set.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
