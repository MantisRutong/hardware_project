#!/usr/bin/env python3
"""Re-run ORB-SLAM3 over an ALREADY-RECORDED episode, from the IR frames and
IMU samples on disk, and write the trajectory it produces -- the offline half
of what synced_capture.py does live.

## Why this exists

synced_capture.py tracks while it records, which is what the teleoperation
branch needs (umi_teleop.py drives the arm from the live pose, so there is
nothing to be offline about). The data-collection branch does not: its
trajectory is a training label, read long after the fact. Recording and
tracking are tied together there only because they happened to be written
that way, and that coupling costs three things:

1. RATE. Live tracking is throttled to 10Hz (OrbSlamWorker.min_interval --
   20Hz reproduced a crash that took several rounds of live testing to
   root-cause). The camera records at ~30fps, so roughly two thirds of the
   recorded frames never get a pose and are dropped by export_dataset.py,
   which is pose-driven. Offline there is no capture thread to starve and
   no real-time deadline at all, so there is no throttle: --min-interval
   defaults to 0, and what paces tracking instead is --min-imu-per-frame,
   the condition that actually has to hold (see its comment in
   replay_episode -- these recordings stamp frames in pairs, and the guard
   collapses each pair, landing on the camera's real ~30Hz).

   Measured end to end on scan_0011, against maps/workspace: live gave 582
   tracked poses which export_dataset.py turned into 581 frames across 4
   episodes (3 of them speed splits); replay gave 1614 tracked poses and
   1557 exported frames in ONE episode with zero splits. Same recording,
   2.7x the frames, and 3.0x the usable 16-frame training windows.

2. RETROACTIVITY. A recording is tracked once, with whatever atlas, mask and
   settings existed that day. scan_0006-0009 were recorded before
   build_atlas_map.py existed and cold-started their maps from a careful
   pick-and-place, which is the exact failure the atlas was built to fix
   (33-35% of frames tracked, 66-100 resets each). Their raw IR and IMU are
   intact -- camera/gripper_mask.py deliberately keeps the saved PNGs
   unmasked so a later pass can redo this differently -- so they can simply
   be re-tracked against maps/, without the camera.

3. EXPERIMENTS. Comparing --gripper-mask against --no-gripper-mask live
   means two recordings, which differ in how the operator moved as well as
   in the mask. Here it is the same bytes twice, so the mask is the only
   variable.

## What it produces

Per episode, in the episode's own directory (or --output-dir):

  camera_trajectory_replay.csv   what the tracker reported frame by frame
                                 during the replay -- the direct analogue of
                                 camera_trajectory.csv, same schema, kept for
                                 comparison against the live run
  camera_trajectory_replay.tum   ORB-SLAM3's own SaveTrajectoryTUM dump
  camera_trajectory_refined.csv  the one to actually use: every frame
                                 re-resolved against the FINAL map

The refined file is the point (see write_refined_camera_trajectory_csv in
synced_capture.py, which this reuses rather than reimplements). Note it is
written under the name export_dataset.py already prefers, so a replayed
episode feeds the existing export with no flags -- and OVERWRITES any
refined file already there, which is intended: this script is the thing
that generates them.

camera_trajectory.csv is never touched. It is the live record of what
happened during the recording, and stays that.

## Timestamps, and why they are host_time_ns

Frames are fed at frames.csv's host_time_ns / 1e9 and IMU at the accel
sample's own host_time_ns / 1e9 -- NOT either stream's RealSense hardware
timestamp. This is not a choice made here; it mirrors synced_capture.py
exactly (see the long comment in its _on_imu_sample accel branch: with IR
streams running alongside color and IMU, gyro/accel hardware timestamps
were measured advancing about twice as fast as color's, so the streams are
not reliably on one clock domain, while the host wall clock is shared by
construction). Matching it is what makes a replayed trajectory directly
comparable to the live one, row for row.

IMU pairing likewise mirrors the live path: one ORB-SLAM3 IMU sample per
ACCEL sample, carrying the most recent gyro reading. ORB-SLAM3 wants
already-paired 6-DOF samples, the D435i produces two independent streams at
different rates, and this is where the live code resolves that.

## Usage

    # re-track a recording against a pre-built atlas, every frame
    python3 output_script/replay_slam.py recording/scan_0009 --map-dir maps/workspace

    # a whole batch (any dir containing scan_* subdirs with frames.csv)
    python3 output_script/replay_slam.py recording --map-dir maps/workspace

    # reproduce what the live run did, for comparison
    python3 output_script/replay_slam.py recording/scan_0011 --map-dir maps/workspace \
        --min-interval 0.1

Run it under this repo's .venv, not conda's Python -- see
camera/orbslam_bridge.py's docstring for why (an RPATH conflict that breaks
any ctypes-loaded .so on this machine).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "camera"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gripper_mask import load_stereo_masks  # noqa: E402
from orbslam_bridge import (  # noqa: E402
    DEFAULT_LIB_PATH as ORBSLAM_DEFAULT_LIB_PATH,
    DEFAULT_SETTINGS_PATH as ORBSLAM_DEFAULT_SETTINGS_PATH,
    DEFAULT_VOCAB_PATH as ORBSLAM_DEFAULT_VOCAB_PATH,
    OrbSlamTracker,
)

# Reused, not reimplemented: these are the live pipeline's own writers, and
# a replayed trajectory has to be byte-comparable with a live one for any of
# the comparisons above to mean anything. Importing synced_capture pulls in
# camera_collecting and spring_position_mode, i.e. the RealSense and Dynamixel
# SDKs -- import-only, no device is opened, so this works with nothing plugged
# in.
from synced_capture import (  # noqa: E402
    ORBSLAM_DEFAULT_RELOCALIZE_SETTINGS_PATH,
    read_tum_trajectory,
    write_camera_trajectory_csv,
    write_refined_camera_trajectory_csv,
)
# One threshold for "a hand carrying a camera never moves this fast", shared
# with the exporter and, through it, with umi_teleop.py's POSE_JUMP_MPS.
from export_dataset import DEFAULT_MAX_SPEED_MPS, _step_between  # noqa: E402


def load_frame_index(episode_dir: Path) -> list[dict[str, Any]]:
    """One entry per recorded frame: host timestamp plus absolute paths to
    both IR images.

    From frames.csv rather than ir_left.txt/ir_right.txt: those index files
    carry the RealSense stream timestamp, while frames.csv is the row that
    also holds host_time_ns -- the clock the live tracker actually fed (see
    the module docstring). Rows without both IR paths are skipped: IR is an
    optional stream (camera_collecting.py's record_ir), so a recording made
    without it simply has nothing to replay."""
    frames_csv = episode_dir / "frames.csv"
    if not frames_csv.is_file():
        return []
    out: list[dict[str, Any]] = []
    with frames_csv.open() as f:
        for row in csv.DictReader(f):
            left, right = row.get("ir_left_path"), row.get("ir_right_path")
            if not left or not right:
                continue
            out.append({
                "timestamp": int(row["host_time_ns"]) / 1e9,
                "left": episode_dir / left,
                "right": episode_dir / right,
            })
    out.sort(key=lambda r: r["timestamp"])
    return out


def load_imu_samples(episode_dir: Path) -> list[tuple[float, float, float, float, float, float, float]]:
    """(timestamp, gx, gy, gz, ax, ay, az), one per ACCEL sample, each
    carrying the most recent gyro reading at or before it.

    Exactly what _on_imu_sample does live, replayed from disk: accel is the
    trigger, gyro is held. Accel samples arriving before any gyro are
    dropped, because live they were dropped too (the `have_gyro` guard) --
    with no gyro yet there is no sample to build. On this rig that is a
    handful of samples at the very start."""
    def read(name: str) -> list[tuple[float, float, float, float]]:
        path = episode_dir / "imu" / name
        if not path.is_file():
            return []
        with path.open() as f:
            rows = [(int(r["host_time_ns"]) / 1e9, float(r["x"]), float(r["y"]), float(r["z"]))
                    for r in csv.DictReader(f)]
        rows.sort(key=lambda r: r[0])
        return rows

    accel, gyro = read("accel.csv"), read("gyro.csv")
    samples = []
    gi = 0
    last_gyro: tuple[float, float, float] | None = None
    for ts, ax, ay, az in accel:
        while gi < len(gyro) and gyro[gi][0] <= ts:
            last_gyro = gyro[gi][1:]
            gi += 1
        if last_gyro is None:
            continue
        samples.append((ts, last_gyro[0], last_gyro[1], last_gyro[2], ax, ay, az))
    return samples


def construct_tracker(args: argparse.Namespace) -> OrbSlamTracker:
    """Same os.chdir() dance as synced_capture.py's construct_orb_tracker,
    and for the same reason: ORB-SLAM3 hardcodes the atlas load path as
    ./atlas.osa relative to the process's cwd AT CONSTRUCTION TIME (see
    System::LoadAtlas). Safe here with no caveats at all -- this script is
    single-threaded and nothing else is doing relative-path I/O."""
    if args.map_dir is None:
        return OrbSlamTracker(args.settings, args.vocab, args.lib, use_viewer=args.viewer)
    original_cwd = Path.cwd()
    try:
        os.chdir(args.map_dir)
        return OrbSlamTracker(args.relocalize_settings, args.vocab, args.lib, use_viewer=args.viewer)
    finally:
        os.chdir(original_cwd)


def replay_episode(episode_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Tracks one recorded episode start to finish and writes its outputs.
    Returns a summary dict."""
    frames = load_frame_index(episode_dir)
    if not frames:
        return {"episode_dir": str(episode_dir), "status": "skipped_no_ir_frames"}
    imu = load_imu_samples(episode_dir)
    if not imu:
        return {"episode_dir": str(episode_dir), "status": "skipped_no_imu"}

    output_dir = args.output_dir or episode_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    masks = load_stereo_masks() if args.gripper_mask else None
    if args.max_frames:
        frames = frames[:args.max_frames]

    print(f"\n=== {episode_dir} ===")
    print(f"  {len(frames)} frames, {len(imu)} IMU samples, "
          f"{'atlas ' + str(args.map_dir) if args.map_dir else 'cold start'}, "
          f"gripper mask {'on' if masks else 'off'}, "
          f"min-interval {args.min_interval}s")

    tracker = construct_tracker(args)
    trajectory: list[dict[str, Any]] = []
    imu_i = 0
    last_tracked_ts: float | None = None
    resets = 0
    last_epoch: int | None = None
    unreadable = 0
    skipped_no_imu = 0
    imu_since_last_frame = 0
    # Did tracking ever reach the LOADED atlas, and when? Loading a map does
    # not put tracking into it: the session cold-starts its own map and stays
    # there until LoopClosing recognises the region and Atlas::ChangeMap
    # switches over (see OrbSlamTracker.current_map_id). A merge is "the map
    # id changed while map_epoch did NOT" -- a reset changes the id too, but
    # bumps the epoch, so tracking each session map against its own epoch
    # keeps the two apart. None means it never happened, which is a result
    # worth reporting rather than an absence: it means the atlas was loaded
    # and not used.
    merged_at_s: float | None = None
    session_map_id: int | None = None
    started = time.monotonic()
    try:
        for i, frame in enumerate(frames):
            ts = frame["timestamp"]
            # Every IMU sample up to this frame, before the frame -- ORB-SLAM3
            # drains its buffer inside track_stereo (see orb_capi.h), so a
            # sample fed afterwards would land in the NEXT frame's interval.
            n_new_imu = 0
            while imu_i < len(imu) and imu[imu_i][0] <= ts:
                tracker.feed_imu(*imu[imu_i])
                imu_i += 1
                n_new_imu += 1
            imu_since_last_frame += n_new_imu
            if args.min_interval > 0 and last_tracked_ts is not None \
                    and ts - last_tracked_ts < args.min_interval:
                continue
            # Stereo-Inertial integrates IMU over the interval BETWEEN two
            # tracked frames (Tracking::PreintegrateIMU). Give it an
            # interval with too little in it and there is nothing to
            # preintegrate -- "not IMU meas" in StereoInitialization at
            # best, a segfault once past it.
            #
            # This guard is what makes full-rate replay possible at all, and
            # it exists because of a real property of these recordings:
            # frames.csv's host_time_ns arrives in PAIRS. On scan_0011's
            # first 400 rows, 183 consecutive pairs are under 1ms apart and
            # 210 are 20-40ms apart -- i.e. two rows stamped almost
            # together, then a normal 30fps gap. The images themselves are
            # all distinct (1887 unique files, 1887 unique stream
            # timestamps); it is the HOST stamp that clusters, because it is
            # taken when the frameset is drained, not when it was exposed.
            # Accel runs at ~200Hz (~5ms), so the sub-millisecond half of
            # each pair spans no IMU at all.
            #
            # Live, the 10Hz throttle stepped over these pairs and they were
            # invisible. Replaying every row walks straight into them:
            # ORB-SLAM3 segfaulted right after "New Map created" on the
            # first attempt here. Measured boundary on scan_0011, with
            # --min-interval 0: 2 samples still crashes, 3 does not.
            #
            # Skipping the second row of a pair costs essentially nothing --
            # it is ~1ms from its neighbour, which IS tracked -- and what
            # comes out is a clean ~30Hz: 211 tracked attempts over 7.0s of
            # those 400 rows, which is the camera's real frame rate.
            #
            # Preferred over a time-based --min-interval because it adapts
            # to the IMU rate instead of hard-coding an assumption about it,
            # and because the thing that actually has to be true is about
            # IMU samples, not about milliseconds.
            if imu_since_last_frame < args.min_imu_per_frame:
                skipped_no_imu += 1
                continue
            left = cv2.imread(str(frame["left"]), cv2.IMREAD_GRAYSCALE)
            right = cv2.imread(str(frame["right"]), cv2.IMREAD_GRAYSCALE)
            if left is None or right is None or left.shape != right.shape:
                unreadable += 1
                continue
            if masks is not None:
                left = masks[0].apply(left)
                right = masks[1].apply(right)
            pose = tracker.track_stereo(ts, np.ascontiguousarray(left), np.ascontiguousarray(right))
            last_tracked_ts = ts
            imu_since_last_frame = 0
            epoch = tracker.last_map_epoch
            if last_epoch is not None and epoch != last_epoch:
                resets += 1
            if args.map_dir is not None and merged_at_s is None:
                map_id = tracker.current_map_id
                if session_map_id is None or epoch != last_epoch:
                    session_map_id = map_id      # this epoch's own cold-started map
                elif map_id != session_map_id:
                    merged_at_s = ts - frames[0]["timestamp"]
                    print(f"\n  merged into the loaded atlas at +{merged_at_s:.1f}s "
                          f"(map {session_map_id} -> {map_id})")
            last_epoch = epoch
            if pose is None:
                trajectory.append({"timestamp": ts, "x": 0.0, "y": 0.0, "z": 0.0,
                                   "q_x": 0.0, "q_y": 0.0, "q_z": 0.0, "q_w": 1.0,
                                   "is_lost": 1, "map_epoch": epoch})
            else:
                x, y, z, qx, qy, qz, qw = pose
                trajectory.append({"timestamp": ts, "x": x, "y": y, "z": z,
                                   "q_x": qx, "q_y": qy, "q_z": qz, "q_w": qw,
                                   "is_lost": 0, "map_epoch": epoch})
            if i % 200 == 0:
                n_ok = sum(1 for r in trajectory if r["is_lost"] == 0)
                print(f"\r  frame {i + 1}/{len(frames)}  tracked {n_ok}  resets {resets}", end="", flush=True)
        print(f"\r  frame {len(frames)}/{len(frames)}" + " " * 30)

        n_tracked = sum(1 for r in trajectory if r["is_lost"] == 0)
        write_camera_trajectory_csv(output_dir / "camera_trajectory_replay.csv", trajectory)

        summary: dict[str, Any] = {
            "episode_dir": str(episode_dir),
            "status": "ok",
            "frames_total": len(trajectory),
            "frames_tracked": n_tracked,
            "map_resets": resets,
            "unreadable_frames": unreadable,
            "skipped_no_imu": skipped_no_imu,
            "merged_into_atlas_at_s": merged_at_s,
            "seconds": round(time.monotonic() - started, 1),
        }

        # Guarded on n_tracked for the same reason synced_capture.py guards
        # it: System::SaveTrajectoryTUM opens with vpKFs[0]->GetPoseInverse()
        # on an unchecked vector, so an episode whose map never initialized
        # would take the process down in C++, where no Python except can
        # catch it.
        if n_tracked == 0:
            summary["refined"] = "skipped_nothing_tracked"
            return summary
        tracker.shutdown()

        # n_tracked > 0 is NOT enough. System::SaveTrajectoryTUM opens with
        # vpKFs[0]->GetPoseInverse() on Atlas::GetAllKeyFrames(), which
        # returns the CURRENT map's keyframes, unchecked -- so an episode
        # that tracked fine and then hit a reset just before shutdown hands
        # it an empty vector and takes the process down in C++, where no
        # Python except can catch it. Seen on task1/scan_0007 ("LM: Active
        # map reset ... 2060 Frames set to lost", then Shutdown, then a
        # segfault) and on three of scan_0006-0009 before that; in a batch
        # it also costs every episode queued behind the one that crashed.
        #
        # Read after shutdown(), when the mapping and loop-closing threads
        # have stopped (the wait added to System::Shutdown()), so the count
        # cannot change under us between the check and the call.
        n_kf = tracker.place_recognition_gates["keyframes"]
        if n_kf == 0:
            summary["refined"] = "skipped_final_map_empty"
            print("  refined: skipped -- the final map has no keyframes (a reset landed just before "
                  "shutdown), so there is nothing to re-resolve against. The replay trajectory is "
                  "still written.")
            return summary
        tum_path = output_dir / "camera_trajectory_replay.tum"
        tracker.save_trajectory_tum(tum_path)
        refined_path = output_dir / "camera_trajectory_refined.csv"
        stats = write_refined_camera_trajectory_csv(
            refined_path, trajectory, read_tum_trajectory(tum_path))
        summary["refined"] = stats
        summary["speed_jumps_live"] = count_speed_jumps(trajectory, args.max_speed_mps)
        summary["speed_jumps_refined"] = count_speed_jumps(
            read_trajectory_csv(refined_path), args.max_speed_mps)
        return summary
    finally:
        tracker.close()


def read_trajectory_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def count_speed_jumps(rows: list[dict[str, Any]], max_speed_mps: float) -> int:
    """How many times consecutive tracked poses imply an impossible speed --
    the same test export_dataset.py splits episodes on, run here so a replay
    reports whether it actually fixed anything.

    A correctly refined trajectory should report 0: the coordinate-frame
    shift an atlas merge introduces has been absorbed rather than recorded.
    Compared against the same count on the replay's own live-equivalent
    rows, this is the acceptance criterion for the whole exercise. Only
    within one map_epoch -- across a reset there is no expectation of
    continuity, which is exactly what map_epoch already flags."""
    tracked = [r for r in rows if str(r["is_lost"]) == "0"]
    jumps = 0
    for prev, row in zip(tracked, tracked[1:]):
        if str(prev.get("map_epoch")) != str(row.get("map_epoch")):
            continue
        step = _step_between({k: str(v) for k, v in prev.items()},
                             {k: str(v) for k, v in row.items()})
        if step is not None and step[2] > max_speed_mps:
            jumps += 1
    return jumps


def expand_episode_dirs(inputs: list[Path]) -> list[Path]:
    """Each input is either an episode dir (has frames.csv) or a parent to
    glob scan_* under. Keyed on frames.csv rather than camera_trajectory.csv
    -- unlike export_dataset.py, this script does not need the episode to
    have been tracked before, which is the entire point for scan_0006-0009,
    and it also lets a mapping pass's own recording (build_atlas_map.py's
    --output-dir) be replayed."""
    out: list[Path] = []
    for item in inputs:
        if (item / "frames.csv").is_file():
            out.append(item)
            continue
        out.extend(sorted(d for d in item.glob("scan_*") if (d / "frames.csv").is_file()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", type=Path, nargs="+",
                         help="Episode dirs (containing frames.csv) and/or parent dirs to glob scan_* under")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Where to write the trajectories (default: alongside each episode's own data, "
                              "which is where export_dataset.py looks for them)")
    parser.add_argument("--map-dir", type=Path, default=None,
                         help="Localize against the atlas.osa in this directory (see build_atlas_map.py) "
                              "instead of cold-starting a map. This is the point of replaying a recording "
                              "made before the atlas existed.")
    parser.add_argument("--min-interval", type=float, default=0.0,
                         help="Seconds between tracked frames; 0 (default) tracks every recorded frame. "
                              "Live capture is pinned to 0.1 because tracking competes with the capture "
                              "thread there -- offline nothing does, so the throttle has no reason to "
                              "exist. Set 0.1 to reproduce a live run for comparison.")
    parser.add_argument("--gripper-mask", dest="gripper_mask", action=argparse.BooleanOptionalAction,
                         default=True,
                         help="Mask the wrist-mounted gripper out before tracking (default: on, matching "
                              "synced_capture.py). --no-gripper-mask on the same recording is the clean "
                              "A/B the live pipeline cannot do.")
    parser.add_argument("--max-speed-mps", type=float, default=DEFAULT_MAX_SPEED_MPS,
                         help=f"Report (do not modify) steps faster than this as coordinate-frame jumps "
                              f"(default: {DEFAULT_MAX_SPEED_MPS} m/s). A correct refined trajectory should "
                              f"report zero.")
    parser.add_argument("--min-imu-per-frame", type=int, default=3,
                         help="Skip a frame unless at least this many IMU samples arrived since the last "
                              "tracked one (default: 3, the measured floor -- 2 still segfaults). This rig "
                              "stamps frames in near-simultaneous pairs, so without it a replay hits "
                              "intervals containing no IMU at all. 0 disables the guard.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (for a quick trial)")
    parser.add_argument("--settings", type=Path, default=ORBSLAM_DEFAULT_SETTINGS_PATH)
    parser.add_argument("--relocalize-settings", type=Path, default=ORBSLAM_DEFAULT_RELOCALIZE_SETTINGS_PATH)
    parser.add_argument("--vocab", type=Path, default=ORBSLAM_DEFAULT_VOCAB_PATH)
    parser.add_argument("--lib", type=Path, default=ORBSLAM_DEFAULT_LIB_PATH)
    parser.add_argument("--viewer", dest="viewer", action=argparse.BooleanOptionalAction, default=False,
                         help="Show ORB-SLAM3's Pangolin viewer while replaying (default: off)")
    args = parser.parse_args()

    # Resolved before any chdir into --map-dir (construct_tracker), which
    # would otherwise re-root every relative path passed on the command line.
    args.inputs = [p.resolve() for p in args.inputs]
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
    if args.map_dir is not None:
        args.map_dir = args.map_dir.resolve()

    episodes = expand_episode_dirs(args.inputs)
    if not episodes:
        raise SystemExit("No episodes found (no frames.csv under any input path)")

    summaries = [replay_episode(e, args) for e in episodes]

    print("\n=== summary ===")
    for s in summaries:
        if s["status"] != "ok":
            print(f"  {Path(s['episode_dir']).name}: {s['status']}")
            continue
        pct = 100.0 * s["frames_tracked"] / max(1, s["frames_total"])
        line = (f"  {Path(s['episode_dir']).name}: tracked {s['frames_tracked']}/{s['frames_total']} "
                f"({pct:.0f}%), resets {s['map_resets']}, {s['seconds']}s")
        if args.map_dir is not None:
            m = s.get("merged_into_atlas_at_s")
            line += f", atlas merge {'+%.1fs' % m if m is not None else 'NEVER'}"
        if isinstance(s.get("refined"), dict):
            line += (f", refined {s['refined']['refined']}/{s['refined']['rows']}"
                     f", jumps {s['speed_jumps_live']} -> {s['speed_jumps_refined']}")
        else:
            line += f", refined: {s['refined']}"
        print(line)


if __name__ == "__main__":
    main()
