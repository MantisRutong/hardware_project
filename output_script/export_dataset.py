"""Convert synced_capture.py's raw per-episode recordings into a Diffusion
Policy / UMI-style replay buffer (a single Zarr store), ready for training.

Why this format specifically: this project's earlier reference work at
~/projects/diffusion_policy already has a working training pipeline built
around this exact convention -- see its
third_party/universal_manipulation_interface/scripts_slam_pipeline/
07_generate_replay_buffer.py, the canonical/most complete example found
there. This script reimplements the same replay-buffer schema (array names,
per-episode bookkeeping) SELF-CONTAINED in this repo -- it does not import
anything from that other project's location on disk, since this repo is
meant to stand on its own (see this project's own history: ORB-SLAM3 was
deliberately vendored into this folder rather than referenced from
~/projects/diffusion_policy for the same reason).

Output Zarr layout (Zarr format 2, a plain directory store):
  data/
    robot0_eef_pos             (N, 3)    f32  -- ABSOLUTE position, meters, per-episode origin (safe
                                                  despite live resets: every episode is already one
                                                  map_epoch segment, see WHY SPLIT below -- no episode
                                                  ever spans a reset). Present so this dataset is
                                                  directly loadable by the stock UMI dataset code,
                                                  which expects this exact key, absolute.
    robot0_eef_rot_axis_angle  (N, 3)    f32  -- ABSOLUTE orientation, axis-angle, same per-episode-
                                                  origin safety as robot0_eef_pos above
    robot0_eef_pos_delta       (N, 3)    f32  -- position CHANGE from the previous row, meters,
                                                  WORLD frame (same frame demo_start_pose/
                                                  demo_end_pose are in); row 0 of every episode
                                                  is (0,0,0) -- no previous row to diff against
    robot0_eef_rot_axis_angle_delta
                                (N, 3)    f32  -- orientation CHANGE from the previous row, as a
                                                  rotation vector, BODY frame (rot_prev^-1 * rot_curr,
                                                  not rot_curr * rot_prev^-1 -- see ROTATION
                                                  REPRESENTATION); row 0 of every episode is (0,0,0)
    robot0_gripper_width      (N, 1)    f32  -- meters (converted from gripper_width_mm)
    robot0_demo_start_pose    (N, 6)    f32  -- (pos3, rotvec3) of this segment's FIRST frame (ABSOLUTE,
                                                  not delta -- an anchor, see POSITION REPRESENTATION),
                                                  repeated every row
    robot0_demo_end_pose      (N, 6)    f32  -- (pos3, rotvec3) of this segment's LAST frame (ABSOLUTE),
                                                  repeated every row
    camera0_rgb                (N, H, W, 3) u1  -- RGB, native recording resolution
  meta/
    episode_ends               (num_episodes,) i8  -- cumulative row count at the end of each episode

"Episode" here means one stretch of trajectory known to be in a SINGLE
coordinate frame, not one recorded scan_XXXX folder. Two things end such a
stretch, and both get their own section below: a map_epoch change (a reset,
which ORB-SLAM3 reports) and an impossible-speed step (a merge, which it
does not).

WHY SPLIT AT map_epoch, NOT JUST AT RECORDING BOUNDARIES:
ORB-SLAM3 resets live during recording (confirmed repeatedly in this
project's live testing -- not just at startup), and each reset moves the
active map's pose origin to an arbitrary new point with no relation to the
previous one (see camera_trajectory.csv's map_epoch column and
write_camera_trajectory_csv's docstring in synced_capture.py for the full
story). Treating a whole scan_XXXX recording as one continuous trajectory
when it actually contains 2-3 disjoint coordinate frames spliced together
would teach the policy an invisible teleport across the reset boundary.
Splitting at every map_epoch change is the fix discussed and settled on
earlier in this project's development.

Switching position/rotation to delta encoding (see below) does NOT remove
the need for this split -- it changes the shape of the failure if you
DON'T split, but doesn't eliminate it. Without splitting, the single
frame-to-frame delta computed ACROSS a reset boundary would still be
fabricated: there is no real geometric relationship between the last
pre-reset pose and the first post-reset one, so nothing correct can be
written there -- not zero (that falsely claims no motion happened across
that instant), not the raw difference (physically meaningless, since the
two poses live in unrelated coordinate frames). Diffusion policy trains on
WINDOWS of consecutive frames, not single steps, so one fabricated frame
sitting inside an otherwise-real window risks corrupting every training
sample that happens to include it -- a smaller blast radius than the old
"entire trailing sub-trajectory is wrong" absolute-position failure, but
still a real, avoidable one. Splitting into separate replay-buffer episodes
(exactly what episode_ends exists to support -- the standard mechanism
sequence-model dataset loaders use to guarantee no sampled window spans two
different demonstrations) avoids this cleanly: no window can span a reset
because none can span an episode boundary. So the split stays, unconditionally.

WHY ALSO SPLIT AT A PHYSICALLY IMPOSSIBLE SPEED:
map_epoch catches every RESET, but not every origin move. Once an atlas is
loaded (synced_capture.py --orbslam-map-dir), ORB-SLAM3 can MERGE the
session's own map into the loaded one: LoopClosing rigidly transforms every
keyframe and map point into the atlas's frame (ApplyScaledRotation) and
switches the active map (Atlas::ChangeMap). Neither touches Atlas's reset
counter -- both of its increment sites (Atlas.cc, in CreateNewMap and
clearMap) are reset paths -- so map_epoch does NOT change across a merge
even though the pose origin just moved. Nothing else in the row changes
either: is_lost stays 0 (tracking really is fine), timestamps stay
continuous, the RGB match stays exact. Every column reads healthy because
every column IS healthy; what moved is which coordinate frame the numbers
are expressed in, and no column records that.

What it looks like in real data (recording/scan_0011, recorded against
maps/workspace): inside ONE map_epoch segment, median frame-to-frame motion
1.4cm, with three steps of 23.3cm, 38.2cm and 40.4cm -- 7.0, 10.9 and
4.0 m/s. A hand carrying a camera does not reach 10.9 m/s, and there are no
intermediate values: 1.4cm, then 38cm, then 1.4cm again. That is a step, not
the tail of a noise distribution.

So speed is used as the detector map_epoch cannot be: above max_speed_mps,
the origin moved, whatever the other columns say. The threshold's default
(2.0 m/s) is umi_teleop.py's POSE_JUMP_MPS, deliberately the same number --
"a hand carrying a camera never exceeds this" is one claim about the
hardware, and it should not be asserted twice with two different values.
Until now only the teleoperation branch acted on it, so an identical bad
pose was rejected before reaching the arm but accepted as a training label.

The response is to SPLIT, not to smooth or clamp. Poses on both sides of the
step are individually correct -- each is right in its own frame -- so
smoothing would blend two correct numbers into a wrong one, and clamping
would fabricate 40cm of motion the camera never made. What is actually true
is "two self-consistent trajectories with no known relation between them",
and a segment boundary is how this format says exactly that. It also
restores the invariant the absolute-pose arrays already rely on (see
POSITION AND ROTATION REPRESENTATION): one episode, one coordinate frame.

Measured cost on scan_0011: 2 segments/582 frames -> 5 segments/581 frames.
One frame, because the jumps are rare and land mid-segment, so splitting
yields large pieces rather than fragments. A recording where this splits
into many too-short segments is telling you the trajectory is unusable, not
that the threshold is wrong.

The real FIX for a merge is camera_trajectory_refined.csv (see
camera_trajectory_path below and write_refined_camera_trajectory_csv in
synced_capture.py), which re-resolves every frame against the final map so
the discontinuity is absorbed rather than recorded -- no split needed and no
frame lost. This check stays regardless: it is what verifies that worked
(a refined episode should report zero splits), and it is the last thing
standing between an undetected origin move and a trained policy.


WHY ONE ROW PER TRACKED POSE, NOT ONE PER CAMERA FRAME:
camera_trajectory.csv only has a pose for frames ORB-SLAM3 actually
attempted (throttled to 10Hz -- see OrbSlamWorker in synced_capture.py),
sparser than the ~30fps camera stream. Rather than interpolating a
synthetic pose for every camera frame, this script keeps one row per REAL
tracked pose and pulls in the nearest-timestamp RGB frame for it -- the
same nearest-match approach (with a rejectable offset threshold) used by
~/projects/diffusion_policy's own build_umi_scene_dataset.py. This lands
the exported dataset at roughly a 10Hz control rate, which is in the same
ballpark as UMI's own reported training/control frequency, not a
compromise made purely for this project's convenience.

POSITION AND ROTATION REPRESENTATION: BOTH absolute pose
(robot0_eef_pos, robot0_eef_rot_axis_angle -- matching upstream's own
replay-buffer convention exactly, see 07_generate_replay_buffer.py) AND
frame-to-frame CHANGE (robot0_eef_pos_delta, robot0_eef_rot_axis_angle_delta
-- an explicit project requirement) are stored side by side. This was
originally delta-only, which technically worked but meant the stock UMI
dataset loader (which reads robot0_eef_pos/robot0_eef_rot_axis_angle by
name, absolute) couldn't load this data without a custom loader being
written. Storing both costs almost nothing (lowdim arrays are tiny next to
the image data) and removes that requirement -- this data now loads
directly under the existing pipeline's expected schema, while also
carrying the delta fields for anything that wants them directly instead of
computing them from the absolute fields.

Storing absolute pose per-episode is safe DESPITE ORB-SLAM3's live resets
specifically because of the map_epoch split below: every episode here is
already one contiguous segment between resets, so "absolute" always means
"absolute within this one episode's single consistent coordinate frame" --
never spanning a reset, same guarantee demo_start_pose/demo_end_pose
(themselves absolute, and always have been) already relied on. This isn't
a new exception to the reset problem -- it's the same fix (the split)
applied to two more fields.

demo_start_pose/demo_end_pose remain useful even with robot0_eef_pos
present directly: they're explicit anchors that don't require scanning to
row 0/-1 of an episode to find the boundary values, and (together with the
delta fields) let the full absolute trajectory be reconstructed even if
robot0_eef_pos/robot0_eef_rot_axis_angle were ever dropped from a
downstream copy of this data -- position via demo_start_pose[:3] +
cumsum(robot0_eef_pos_delta), rotation by sequentially composing (not
summing -- rotation isn't a vector space) robot0_eef_rot_axis_angle_delta
onto demo_start_pose[3:].

ROTATION REPRESENTATION DETAIL: robot0_eef_rot_axis_angle_delta uses the
BODY frame (rot_prev^-1 * rot_curr), not world frame (rot_curr *
rot_prev^-1) -- matching this project's own reference precedent
(~/projects/diffusion_policy's tool_imu_pose.zarr documents its action as
"drotation_vector_body"). Body-frame deltas describe how much the gripper
rotated about ITS OWN axes, independent of which way it happened to be
facing in the world at that instant -- the physically natural quantity for
an action a policy would actually issue, and consistent with position
delta being stored in world frame (position and rotation deltas are not
expected to share a frame convention here; each uses whichever convention
is standard for that quantity).

Usage:
    export_dataset.py recording/scan_0001 recording/scan_0002 --output data/replay_buffer.zarr
    export_dataset.py recording --output data/replay_buffer.zarr   # expands to recording/scan_*
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numcodecs
import numpy as np
import zarr
from scipy.spatial.transform import Rotation


# umi_teleop.py's POSE_JUMP_MPS, on purpose -- same physical claim about the
# same hardware, so it gets one value, not two. See WHY ALSO SPLIT AT A
# PHYSICALLY IMPOSSIBLE SPEED in the module docstring.
DEFAULT_MAX_SPEED_MPS = 2.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def camera_trajectory_path(episode_dir: Path) -> Path:
    """Prefer the refined trajectory when the episode has one.

    camera_trajectory.csv is what ORB-SLAM3 reported live, frame by frame.
    camera_trajectory_refined.csv is the same episode re-resolved from its
    FINAL map after shutdown (see write_refined_camera_trajectory_csv in
    synced_capture.py). The refined one is strictly better as a training
    label: it has the same schema and the same map_epoch column, but the
    coordinate-frame shift an atlas merge introduces mid-episode -- which
    map_epoch cannot flag, because merging is not a reset -- has been
    absorbed instead of recorded as a teleport.

    Falls back to the live file, so episodes recorded before this existed
    (and any where the refinement step failed) still export. With live
    tracking off (synced_capture.py's default now) there IS no live file
    and the refined one is the only trajectory the episode has -- see
    has_trajectory above."""
    refined = episode_dir / "camera_trajectory_refined.csv"
    return refined if refined.is_file() else episode_dir / "camera_trajectory.csv"


def has_trajectory(episode_dir: Path) -> bool:
    """Whether this directory has a trajectory to export at all.

    Either file counts. A recording made with live tracking has
    camera_trajectory.csv; one recorded with tracking off (the default now
    -- see synced_capture.py's --orbslam) has only what replay_slam.py
    wrote afterwards, and that is camera_trajectory_refined.csv. Requiring
    the live file would silently skip every episode produced by the
    offline flow."""
    return ((episode_dir / "camera_trajectory.csv").is_file()
            or (episode_dir / "camera_trajectory_refined.csv").is_file())


def expand_episode_dirs(inputs: list[Path]) -> list[Path]:
    """Each input is either an episode dir (has a trajectory file directly)
    or a parent dir to glob scan_* episode dirs beneath."""
    episodes: list[Path] = []
    for p in inputs:
        if has_trajectory(p):
            episodes.append(p)
        else:
            found = sorted(d for d in p.glob("scan_*") if has_trajectory(d))
            if not found:
                print(f"WARNING: {p} is neither an episode dir nor a parent of scan_* episode dirs -- skipping",
                      file=sys.stderr)
            episodes.extend(found)
    return episodes


def load_synced_index(episode_dir: Path) -> tuple[np.ndarray, list[dict[str, str]]]:
    """synced.csv sorted by host_time_ns (should already be in order, but
    don't assume), for nearest-timestamp lookups against tracked poses."""
    rows = read_csv(episode_dir / "synced.csv")
    rows.sort(key=lambda r: int(r["host_time_ns"]))
    host_ns = np.array([int(r["host_time_ns"]) for r in rows], dtype=np.int64)
    return host_ns, rows


def nearest_synced_row(host_ns: np.ndarray, rows: list[dict[str, str]], query_ns: int) -> tuple[dict[str, str], float]:
    """Returns (matched row, offset_ms). O(log n) per call via searchsorted --
    same approach as build_umi_scene_dataset.py's nearest_indices, just
    single-query instead of vectorized (episode sizes here are small enough
    that this isn't a bottleneck)."""
    right = int(np.clip(np.searchsorted(host_ns, query_ns), 0, len(host_ns) - 1))
    left = max(right - 1, 0)
    if abs(query_ns - host_ns[left]) <= abs(host_ns[right] - query_ns):
        idx = left
    else:
        idx = right
    offset_ms = abs(host_ns[idx] - query_ns) / 1e6
    return rows[idx], offset_ms


def _epoch_of(row: dict[str, str]) -> str:
    # Older recordings (from before this column existed, or from the brief
    # window it was misnamed map_id -- see this project's own history)
    # don't have map_epoch at all. Treat those as one implicit epoch for
    # the whole file rather than crashing -- correct for genuinely reset-
    # free old recordings, silently wrong (no split) for old ones that
    # actually did reset, which is the best available behavior without the
    # data the fix didn't exist yet to record. Recordings made after the
    # map_epoch fix always have it, and won't hit this fallback.
    return row.get("map_epoch") or row.get("map_id") or "0"


def _step_between(prev: dict[str, str], row: dict[str, str]) -> tuple[float, float, float] | None:
    """(step_m, dt_ms, speed_mps) between two consecutive tracked rows, or
    None if the speed cannot be computed.

    None (rather than 0.0 or inf) when dt <= 0, which means duplicate or
    out-of-order timestamps. That is a different defect from the one this
    detector is for, and neither answer would be honest: 0.0 asserts the
    step is fine, inf asserts an origin move that may not have happened.
    None leaves the pair unsplit and lets whatever produced the bad
    timestamps be found on its own terms."""
    dt = float(row["timestamp"]) - float(prev["timestamp"])
    if dt <= 0:
        return None
    step = math.dist(
        (float(prev["x"]), float(prev["y"]), float(prev["z"])),
        (float(row["x"]), float(row["y"]), float(row["z"])),
    )
    return step, dt * 1000, step / dt


def segment_tracked_rows(
    trajectory_rows: list[dict[str, str]],
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
) -> tuple[list[list[dict[str, str]]], list[dict[str, Any]]]:
    """Splits tracked (is_lost==0) trajectory rows into contiguous runs that
    share one map_epoch AND contain no impossible-speed step -- see the
    module docstring's two SPLIT sections for both halves of the rule.

    Both cuts mean the same thing (the pose origin moved here, so what
    follows is in a different coordinate frame), which is why they are one
    pass producing one kind of segment rather than a second filter layered
    on top: everything downstream -- min_segment_frames, per-segment
    reports, episode_ends -- then treats a merge exactly as it already
    treats a reset, with no special case.

    Returns (segments, jumps). jumps is one dict per speed cut, for the
    conversion report: a merge leaves no other trace in the data, so if this
    is not reported it is not observable at all. max_speed_mps <= 0 disables
    the speed cut (map_epoch splitting always stays -- see the module
    docstring: "So the split stays, unconditionally")."""
    tracked = [r for r in trajectory_rows if r["is_lost"] == "0"]
    segments: list[list[dict[str, str]]] = []
    jumps: list[dict[str, Any]] = []
    for row in tracked:
        if not segments or _epoch_of(segments[-1][-1]) != _epoch_of(row):
            segments.append([row])
            continue
        prev = segments[-1][-1]
        step = _step_between(prev, row) if max_speed_mps > 0 else None
        if step is not None and step[2] > max_speed_mps:
            step_m, dt_ms, speed_mps = step
            jumps.append({
                "status": "split_impossible_speed",
                "map_epoch": _epoch_of(row),
                "last_timestamp_before_split": prev["timestamp"],
                "first_timestamp_after_split": row["timestamp"],
                "step_m": round(step_m, 4),
                "dt_ms": round(dt_ms, 1),
                "speed_mps": round(speed_mps, 2),
            })
            segments.append([row])
        else:
            segments[-1].append(row)
    return segments, jumps


def process_episode(
    episode_dir: Path,
    min_segment_frames: int,
    max_rgb_offset_ms: float,
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Returns (segments, reports) -- segments is a list of per-episode row
    lists (each inner list is one replay-buffer episode: a surviving
    map_epoch segment, with per-frame dicts eef_pos, eef_rot_axis_angle
    (both absolute), eef_pos_delta, eef_rot_axis_angle_delta, gripper_width_m,
    demo_start_pose, demo_end_pose, image_path -- plus a transient eef_rot
    Rotation object, used only internally for exact delta composition, not
    exported), kept as separate lists rather than flattened so the caller can derive
    episode_ends directly from list boundaries instead of re-deriving them
    heuristically afterward -- an earlier version of this function
    flattened everything and had the caller detect boundaries by comparing
    demo_start_pose between consecutive rows, which silently MERGED
    distinct episodes whenever two segments happened to start at the same
    pose (extremely common: ORB-SLAM3's first tracked frame after any
    reset is almost always exactly (0,0,0), so most segments share a start
    pose by construction, not coincidence). reports is one summary dict per
    segment (including skipped ones) for the conversion report."""
    trajectory_rows = read_csv(camera_trajectory_path(episode_dir))
    host_ns, synced_rows = load_synced_index(episode_dir)
    segments, jumps = segment_tracked_rows(trajectory_rows, max_speed_mps)

    out_segments: list[list[dict[str, Any]]] = []
    # Reported even though the segments they produced are reported too: a
    # merge leaves no other trace, so this is the only place the fact that
    # one happened is written down. Also what tells a refined trajectory
    # (expected: none) from a live one.
    reports: list[dict[str, Any]] = [{"episode_dir": str(episode_dir), **j} for j in jumps]
    for seg_idx, segment in enumerate(segments):
        map_epoch = _epoch_of(segment[0])
        if len(segment) < min_segment_frames:
            reports.append({
                "episode_dir": str(episode_dir), "segment_index": seg_idx, "map_epoch": map_epoch,
                "status": "skipped_too_short", "frames": len(segment),
            })
            continue

        seg_rows: list[dict[str, Any]] = []
        offsets_ms = []
        dropped_missing_image = 0
        for row in segment:
            # camera_trajectory.csv timestamps are host_time_ns/1e9 (seconds)
            # -- see _on_stereo_frame_for_orbslam in synced_capture.py.
            query_ns = round(float(row["timestamp"]) * 1e9)
            synced_row, offset_ms = nearest_synced_row(host_ns, synced_rows, query_ns)
            if offset_ms > max_rgb_offset_ms:
                reports.append({
                    "episode_dir": str(episode_dir), "segment_index": seg_idx, "map_epoch": map_epoch,
                    "status": "dropped_frame_rgb_offset_too_large", "offset_ms": offset_ms,
                    "trajectory_timestamp": row["timestamp"],
                })
                continue
            image_path = episode_dir / synced_row["rgb_path"]
            if not image_path.is_file():
                dropped_missing_image += 1
                continue
            offsets_ms.append(offset_ms)

            pos = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64)
            quat_xyzw = np.array([float(row["q_x"]), float(row["q_y"]), float(row["q_z"]), float(row["q_w"])])
            rotvec = Rotation.from_quat(quat_xyzw).as_rotvec()
            gripper_width_m = float(synced_row["servo_gripper_width_mm"]) / 1000.0

            seg_rows.append({
                "eef_pos": pos.astype(np.float32),
                "eef_rot": Rotation.from_quat(quat_xyzw),  # kept as a Rotation, not rotvec, for exact composition below
                "eef_rot_axis_angle": rotvec.astype(np.float32),
                "gripper_width_m": np.float32(gripper_width_m),
                "image_path": image_path,
            })

        if len(seg_rows) < min_segment_frames:
            reports.append({
                "episode_dir": str(episode_dir), "segment_index": seg_idx, "map_epoch": map_epoch,
                "status": "skipped_too_short_after_rgb_matching", "frames": len(seg_rows),
                "dropped_missing_image": dropped_missing_image,
            })
            continue

        # demo_start_pose/demo_end_pose stay ABSOLUTE (anchors) -- computed
        # from eef_pos BEFORE it's overwritten with deltas below. See
        # POSITION REPRESENTATION in the module docstring.
        start_pose = np.concatenate([seg_rows[0]["eef_pos"], seg_rows[0]["eef_rot_axis_angle"]]).astype(np.float32)
        end_pose = np.concatenate([seg_rows[-1]["eef_pos"], seg_rows[-1]["eef_rot_axis_angle"]]).astype(np.float32)

        # Position/rotation CHANGE from the previous surviving row (not the
        # previous row of the raw segment -- a frame can be dropped above
        # for a too-large RGB offset or a missing image, and the delta
        # should reflect the actual exported sequence, the only one a
        # consumer of this dataset will ever see). Row 0 of the episode has
        # no previous row to diff against -- zero, matching this project's
        # earlier precedent for "no reference yet" (e.g. is_lost placeholder
        # rows). This is also exactly why episodes are split at map_epoch
        # boundaries rather than just at recording boundaries: a delta
        # computed ACROSS a reset would be fabricated (the post-reset pose
        # has no real geometric relationship to the pre-reset one), and
        # unlike row 0's honest "no data yet" zero, a fabricated mid-episode
        # zero would falsely claim the gripper didn't move/rotate across
        # that instant -- see the module docstring's WHY SPLIT section.
        #
        # Rotation delta uses the BODY-frame convention (rot_prev^-1 * rot_curr),
        # matching this project's own reference precedent
        # (~/projects/diffusion_policy's tool_imu_pose.zarr documents its
        # action as "drotation_vector_body") -- NOT world-frame, which would
        # instead be rot_curr * rot_prev^-1. Composed via Rotation objects,
        # not by subtracting rotation vectors (rotation isn't a vector space;
        # subtracting axis-angle representations directly is only a valid
        # approximation for small angles between consecutive frames, not an
        # exact delta in general).
        prev_pos = seg_rows[0]["eef_pos"]
        prev_rot = seg_rows[0]["eef_rot"]
        seg_rows[0]["eef_pos_delta"] = np.zeros(3, dtype=np.float32)
        seg_rows[0]["eef_rot_axis_angle_delta"] = np.zeros(3, dtype=np.float32)
        for r in seg_rows[1:]:
            r["eef_pos_delta"] = (r["eef_pos"] - prev_pos).astype(np.float32)
            r["eef_rot_axis_angle_delta"] = (prev_rot.inv() * r["eef_rot"]).as_rotvec().astype(np.float32)
            prev_pos = r["eef_pos"]
            prev_rot = r["eef_rot"]

        for r in seg_rows:
            r["demo_start_pose"] = start_pose
            r["demo_end_pose"] = end_pose
        out_segments.append(seg_rows)

        reports.append({
            "episode_dir": str(episode_dir), "segment_index": seg_idx, "map_epoch": map_epoch,
            "status": "ok", "frames": len(seg_rows),
            "rgb_offset_ms_mean": float(np.mean(offsets_ms)) if offsets_ms else None,
            "rgb_offset_ms_max": float(np.max(offsets_ms)) if offsets_ms else None,
            "dropped_missing_image": dropped_missing_image,
        })
    return out_segments, reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", type=Path, nargs="+",
                         help="Episode dirs (containing camera_trajectory.csv) and/or parent dirs to glob scan_* under")
    parser.add_argument("--output", type=Path, required=True, help="Output Zarr path (a directory)")
    parser.add_argument("--min-segment-frames", type=int, default=10,
                         help="Drop segments shorter than this many tracked frames, after splitting on "
                              "both map_epoch and --max-speed-mps (default: 10)")
    parser.add_argument("--max-rgb-offset-ms", type=float, default=40.0,
                         help="Drop a frame if the nearest RGB capture is farther than this from the tracked "
                              "pose's timestamp (default: 40ms, a bit over one 30fps frame interval)")
    parser.add_argument("--max-speed-mps", type=float, default=DEFAULT_MAX_SPEED_MPS,
                         help="Split a segment wherever consecutive poses imply a speed above this, which "
                              "means ORB-SLAM3 moved the map origin (an atlas merge) rather than the camera "
                              f"moving (default: {DEFAULT_MAX_SPEED_MPS} m/s, umi_teleop.py's POSE_JUMP_MPS). "
                              "0 disables the check; map_epoch splitting is unconditional either way.")
    args = parser.parse_args()

    episodes = expand_episode_dirs(args.inputs)
    if not episodes:
        raise SystemExit("No episodes found (no camera_trajectory.csv under any input path)")

    all_rows: list[dict[str, Any]] = []
    all_reports: list[dict[str, Any]] = []
    episode_ends: list[int] = []
    # One replay-buffer "episode" per surviving map_epoch segment, which is
    # why episode_ends is built from len(all_rows) transitions below, not
    # from len(episodes) -- see module docstring.
    for episode_dir in episodes:
        segments, reports = process_episode(episode_dir, args.min_segment_frames,
                                            args.max_rgb_offset_ms, args.max_speed_mps)
        all_reports.extend(reports)
        for segment_rows in segments:
            all_rows.extend(segment_rows)
            episode_ends.append(len(all_rows))

    if not all_rows:
        print(json.dumps({"episodes_found": len(episodes), "segments": all_reports}, indent=2))
        raise SystemExit("No usable frames survived processing -- see report above")

    first_img = cv2.imread(str(all_rows[0]["image_path"]))
    if first_img is None:
        raise SystemExit(f"Could not read {all_rows[0]['image_path']}")
    h, w = first_img.shape[:2]
    n = len(all_rows)

    if args.output.exists():
        import shutil
        shutil.rmtree(args.output)
    root = zarr.open_group(str(args.output), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")

    lowdim_compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
    img_compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)

    # Absolute pose, alongside the delta fields below -- safe to store
    # despite ORB-SLAM3's live resets specifically BECAUSE every episode
    # here is already one map_epoch segment (see WHY SPLIT in the module
    # docstring): no episode ever spans a reset, so "absolute" here always
    # means "absolute within this episode's one consistent coordinate
    # frame", exactly like demo_start_pose/demo_end_pose already are. This
    # is what makes the stock UMI dataset loader (which expects these two
    # keys, absolute, un-split scan_XXXX recordings notwithstanding) able
    # to read this data directly instead of needing a custom loader.
    eef_pos = data.create_dataset("robot0_eef_pos", shape=(n, 3), chunks=(min(1000, n), 3),
                                   dtype="f4", compressor=lowdim_compressor)
    eef_rot = data.create_dataset("robot0_eef_rot_axis_angle", shape=(n, 3), chunks=(min(1000, n), 3),
                                   dtype="f4", compressor=lowdim_compressor)
    eef_pos_delta = data.create_dataset("robot0_eef_pos_delta", shape=(n, 3), chunks=(min(1000, n), 3),
                                         dtype="f4", compressor=lowdim_compressor)
    eef_rot_delta = data.create_dataset("robot0_eef_rot_axis_angle_delta", shape=(n, 3), chunks=(min(1000, n), 3),
                                         dtype="f4", compressor=lowdim_compressor)
    gripper_width = data.create_dataset("robot0_gripper_width", shape=(n, 1), chunks=(min(1000, n), 1),
                                         dtype="f4", compressor=lowdim_compressor)
    demo_start = data.create_dataset("robot0_demo_start_pose", shape=(n, 6), chunks=(min(1000, n), 6),
                                      dtype="f4", compressor=lowdim_compressor)
    demo_end = data.create_dataset("robot0_demo_end_pose", shape=(n, 6), chunks=(min(1000, n), 6),
                                    dtype="f4", compressor=lowdim_compressor)
    img_arr = data.create_dataset("camera0_rgb", shape=(n, h, w, 3), chunks=(1, h, w, 3),
                                   dtype="u1", compressor=img_compressor)

    for i, row in enumerate(all_rows):
        eef_pos[i] = row["eef_pos"]
        eef_rot[i] = row["eef_rot_axis_angle"]
        eef_pos_delta[i] = row["eef_pos_delta"]
        eef_rot_delta[i] = row["eef_rot_axis_angle_delta"]
        gripper_width[i] = row["gripper_width_m"]
        demo_start[i] = row["demo_start_pose"]
        demo_end[i] = row["demo_end_pose"]
        img = cv2.imread(str(row["image_path"]))
        if img is None or img.shape != first_img.shape:
            raise SystemExit(f"Invalid/mismatched image at row {i}: {row['image_path']}")
        img_arr[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f"\rwriting frame {i + 1}/{n}", end="", flush=True)
    print()

    # chunks= given explicitly: zarr 2.12's auto-chunk-guessing path calls
    # the long-removed np.product on newer numpy (this venv's), so leaving
    # it to auto-guess raises AttributeError -- harmless-sized array anyway,
    # one chunk is fine.
    meta.array("episode_ends", np.asarray(episode_ends, dtype=np.int64),
               chunks=(max(len(episode_ends), 1),), compressor=None)
    root.attrs.update({
        "pose_representation": "absolute_plus_delta_world_pos_body_rot",
        "gripper_width_units": "meters",
        "action_note": "robot0_eef_pos/robot0_eef_rot_axis_angle are ABSOLUTE, per-episode-origin (matches "
                        "upstream UMI's own replay-buffer convention exactly -- this data loads directly "
                        "under the stock dataset code's expected schema). robot0_eef_pos_delta is "
                        "frame-to-frame position CHANGE in WORLD frame; robot0_eef_rot_axis_angle_delta is "
                        "frame-to-frame rotation CHANGE in BODY frame (rot_prev^-1 * rot_curr) -- an "
                        "additional project-specific requirement stored alongside the absolute fields, not "
                        "instead of them. Row 0 of each episode is (0,0,0) for both delta fields -- no "
                        "previous row to diff against. demo_start_pose/demo_end_pose are ABSOLUTE per-episode "
                        "anchors, redundant with row 0/-1 of robot0_eef_pos/robot0_eef_rot_axis_angle but "
                        "kept as explicit, easy-to-find anchors. See this script's module docstring "
                        "(POSITION AND ROTATION REPRESENTATION) for the full reasoning, including why "
                        "storing absolute pose per-episode is safe despite ORB-SLAM3's live resets (every "
                        "episode is already one contiguous map_epoch segment -- see WHY SPLIT -- so "
                        "'absolute' never spans a reset).",
        "episode_note": "One replay-buffer episode = one contiguous ORB-SLAM3 map_epoch segment, NOT one "
                         "recorded scan_XXXX folder -- a single recording can span multiple episodes here "
                         "if ORB-SLAM3 reset mid-recording.",
    })

    summary = {
        "source_episode_dirs": [str(e) for e in episodes],
        "output": str(args.output),
        "total_frames": n,
        "total_episodes": len(episode_ends),
        "image_shape": [h, w, 3],
        "segment_reports": all_reports,
    }
    (args.output / "conversion_report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "segment_reports"}, indent=2))
    print(f"Wrote {args.output} ({n} frames, {len(episode_ends)} episodes). Full report: "
          f"{args.output / 'conversion_report.json'}")


if __name__ == "__main__":
    main()
