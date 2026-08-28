#!/usr/bin/env python3
"""Drive the xArm live from the handheld UMI gripper's own ORB-SLAM3 pose.

This is the same shape as ~/projects/teleop_collection/scripts/
quest_to_xarm_live.py -- anchor, clutch, deadzone, smoothing, delta clamp,
set_servo_cartesian in servo mode -- with the pose source swapped from a
Quest controller to this project's wrist camera. The control-side constants
below are deliberately copied from that script rather than re-derived: they
were tuned against this same arm and are the one part of this pipeline
already known to behave.

## What is different from the Quest version, and why it matters

**Rate.** OrbSlamWorker is throttled to 10Hz and that number was expensive
to arrive at (see its min_interval comment in synced_capture.py -- 20Hz
reproduced a crash). set_servo_cartesian wants ~50Hz. So the control loop
here runs at 50Hz independently and reads whatever the latest SLAM pose is,
with the existing EMA smoothing bridging the gap. The arm is therefore
always chasing a target up to ~100ms old; that is inherent, not a bug to
tune out, and it is why the delta clamp matters.

**Tracking loss is real here.** A Quest controller essentially never loses
tracking. ORB-SLAM3 does, measurably and not rarely, and a reset MOVES THE
MAP ORIGIN -- the same pose value suddenly means somewhere else. Sending
that to a real arm is a jump, not a glitch. Every path that could produce a
discontinuous pose disengages the clutch and demands a re-anchor:

  - tracking reports lost
  - map_epoch changes (a reset -- see System::GetMapEpoch())
  - no fresh pose within POSE_STALE_S (SLAM stalled, or the camera died)

**Position by default, rotation opt-in (--rotation).** The Quest script
holds the anchor's orientation and commands translation alone; that stays
the default here for a reason. Tracking noise enters orientation directly,
so enabling rotation before position quality is established makes it
impossible to tell which of the two is at fault. Rotation also needs a far
better frame calibration than translation does -- see fit_frame_rotation().

## Safety posture

Dry run is the DEFAULT: without --execute nothing is sent to the arm, and
the intended targets are printed instead. That is the same convention
xarm_policy_rollout.py uses, and it is also how you find --axis-map (below)
without moving anything.

## Finding --axis-map

The camera's frame and the arm's base frame are not related by anything
this script can know -- it depends on how the gripper is held and how the
arm is mounted. The Quest script has the equivalent as a hardcoded R_MAP
found empirically, and this needs the same treatment ONCE:

    python3 umi_teleop.py            # dry run, prints camera + robot deltas
    # engage, move the gripper along one axis at a time, read which robot
    # axis moved and in which direction, then encode it:
    python3 umi_teleop.py --axis-map "+z,-x,-y" --execute

"+z,-x,-y" reads: robot X follows camera +Z, robot Y follows camera -X,
robot Z follows camera -Y. The default "+x,+y,+z" is almost certainly wrong
for any real mounting; it is the identity so that a dry run shows you raw,
unrotated camera motion to reason from.

## Usage

    python3 umi_teleop.py --orbslam-map-dir maps/kitchen_table
    python3 umi_teleop.py --axis-map "+z,-x,-y" --execute

Keys (the clutch -- see the module's design decision): SPACE toggles
engage/disengage, R re-anchors while engaged, Q quits. A toggle rather than
hold-to-move because both hands are on the gripper.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "camera"))
sys.path.insert(0, str(REPO_ROOT / "output_script"))

import camera_collecting as cam  # noqa: E402
from gripper_mask import load_stereo_masks  # noqa: E402
from orbslam_bridge import DEFAULT_LIB_PATH, DEFAULT_VOCAB_PATH, OrbSlamTracker  # noqa: E402
from synced_capture import (  # noqa: E402
    ORBSLAM_DEFAULT_RELOCALIZE_SETTINGS_PATH,
    ORBSLAM_DEFAULT_SETTINGS_PATH,
    OrbSlamWorker,
)

DEFAULT_XARM_IP = "192.168.1.227"
FRAME_FILE = "teleop_frame.json"

# --- Control constants, copied from quest_to_xarm_live.py -------------------
# These were tuned against this same arm. Copied rather than re-derived: the
# pose source changed, the arm's dynamics did not.
# 300 rather than the Quest script's 600: teleoperating from SLAM means the
# pose stream carries tracking noise the Quest's did not, and scaling down
# attenuates that noise on its way to the arm by the same factor. It also
# makes fine positioning easier (10cm of hand -> 3cm of arm). The cost is
# more hand travel per unit of arm travel, which the clutch already handles
# -- disengage, reposition, re-engage, like lifting a mouse.
SCALE_MM_PER_M = 300.0      # hand moves 1m -> arm moves 300mm. 1000 = 1:1.
MAX_CART_DELTA_MM = 80.0    # hard cap on distance from anchor
SPEED = 200                 # reserved by the SDK; kept for parity
MVACC = 1000                # reserved by the SDK; kept for parity
LOOP_DT = 0.02              # 50Hz servo command rate
DEADZONE_M = 0.003
TARGET_ALPHA = 0.55         # higher = more responsive, lower = smoother
STATUS_DT = 0.25

# A pose older than this means SLAM stalled or the camera stopped. At the
# 10Hz submission throttle a healthy stream delivers every ~100ms, so this
# is several missed updates, not one late one.
POSE_STALE_S = 0.5

# --- Two independent guards against a pose jump reaching the arm ---------
#
# MAX_CART_DELTA_MM bounds how FAR the target can be from the anchor. It
# says nothing about how FAST it gets there, and these two guards are what
# cover that gap.
#
# A jump is not hypothetical: a relocalization that lands in the wrong part
# of the map, or a reset that GetMapEpoch() somehow doesn't flag, changes
# what the same pose value means, discontinuously. Smoothing alone does not
# save you -- at TARGET_ALPHA=0.55 an 80mm step in the target still puts
# 44mm into the very next 20ms command, which is ~2.2 m/s of commanded arm
# motion.
#
# First guard, at the source: a hand carrying a camera does not exceed this
# speed. A SLAM jump is typically metres instantaneously, so this catches it
# with enormous margin while never firing on real motion.
POSE_JUMP_MPS = 2.0

# Second guard, at the output: hard cap on how far the COMMANDED pose may
# move per control cycle, whatever the cause. 8mm per 20ms bounds the arm at
# ~0.4 m/s. Chosen to sit well above real teleoperation at the default scale
# (1 m/s of hand -> 300 mm/s of arm -> 6mm per cycle) and well below
# anything that would be alarming to stand next to.
MAX_STEP_MM = 8.0

# How many consecutive rate-limited cycles before treating it as a runaway
# target rather than noise. 10 cycles = 0.2s, during which the arm can have
# moved at most 80mm -- the same envelope MAX_CART_DELTA_MM already allows.
RATE_LIMIT_PATIENCE = 10

# --- Rotation (opt-in, --rotation) --------------------------------------
# The arm's roll/pitch/yaw convention, confirmed empirically against this
# controller rather than taken from the docs (which don't state it): read
# the same pose via get_position() and get_position_aa(), and only
# Rz(yaw) @ Ry(pitch) @ Rx(roll) reproduced the axis-angle rotation, to
# 0.000 degrees. scipy spells that "XYZ" (extrinsic/fixed-axis).
XARM_EULER_SEQ = "XYZ"

# Angular counterparts of MAX_CART_DELTA_MM and MAX_STEP_MM. Rotation gets
# its own limits because the same numeric error hurts more here: a degree
# of wrist error is invisible in the pose readout but clearly visible at
# the fingertips, and tracking noise enters orientation directly.
# Rotation runs at half rate by default. Translation is scaled down for
# noise reasons; rotation is scaled down for a different one -- the camera
# is held in the hand, so wrist articulation produces far more rotation per
# unit of effort than whole-arm translation does, and 1:1 feels twitchy.
ROT_SCALE = 0.5

# Rotations smaller than this are treated as unintended. Translating a
# handheld camera without also rotating it slightly is nearly impossible,
# and without a deadzone that incidental wobble drives the wrist the whole
# time. Subtracted rather than thresholded, so crossing it is continuous
# and there is no jump at the boundary.
ROT_DEADZONE_DEG = 2.0

# Rotation smooths harder than translation (TARGET_ALPHA). Same reason its
# limits are tighter: a degree of wrist error is visible at the fingertips
# where a millimetre of position error is not.
ROT_ALPHA = 0.25

MAX_ROT_DEG = 45.0          # cap on angle away from the anchor orientation
MAX_ROT_STEP_DEG = 1.5      # per 20ms cycle -> 75 deg/s ceiling
POSE_JUMP_DPS = 720.0       # 2 rev/s: far above wrist motion, far below a reset


class QuietStdout:
    """Send ORB-SLAM3's console output to a log file, keeping the terminal
    for this script's own prompts.

    Has to work at the file-descriptor level, not by rebinding sys.stdout:
    the noise comes from C++ inside liborb_capi.so writing to fd 1 directly,
    which no Python-level redirect can intercept. The original fd is duped
    first so prompts can still be written to the real terminal, and the
    noise is kept rather than discarded -- when tracking misbehaves, those
    are the messages that say why."""

    def __init__(self, log_path: Path, enabled: bool = True) -> None:
        self.log_path = log_path
        self.enabled = enabled
        self.term: Any = None
        # Both stdout and stderr: ORB-SLAM3 splits its noise across the two
        # (Settings.cc's "optional parameter does not exist" warnings go to
        # std::cerr, the tracking chatter to std::cout), so redirecting only
        # fd 1 still leaves the terminal cluttered.
        self._saved_fds: dict[int, int] = {}
        self._log: Any = None

    def __enter__(self) -> "QuietStdout":
        if not self.enabled:
            self.term = sys.stdout
            return self
        self.term = os.fdopen(os.dup(1), "w", buffering=1)
        self._log = open(self.log_path, "w", buffering=1)
        for fd in (1, 2):
            self._saved_fds[fd] = os.dup(fd)
            os.dup2(self._log.fileno(), fd)
        return self

    def __exit__(self, *exc: Any) -> None:
        if not self.enabled:
            return
        # Restored in the caller's `finally`, which runs BEFORE an exception
        # propagates -- so a traceback still reaches the real terminal
        # rather than being buried in the log.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        for fd, saved in self._saved_fds.items():
            os.dup2(saved, fd)
            os.close(saved)
        self._saved_fds.clear()
        for f in (self._log, self.term):
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass

    def say(self, text: str = "", end: str = "\n", flush: bool = True) -> None:
        """Write to the real terminal, past the redirect.

        Accepts `flush` purely so it is a drop-in for print() at call sites
        that pass it; output is unbuffered here regardless."""
        self.term.write(text + end)
        self.term.flush()


class _CalibrationDone(Exception):
    """Unwinds out of the `with KeyClutch()` block to main()'s finally, so
    calibration tears the camera and tracker down the same way the control
    loop does instead of duplicating that."""


class PoseSlot:
    """Latest tracked pose, written by the SLAM worker thread and read by
    the control thread. Only ever holds the newest value -- the control loop
    wants "where is the gripper now", never a backlog."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pos: np.ndarray | None = None
        self._rot: Any = None
        self._t = 0.0
        self._epoch: int | None = None
        self._lost_since_read = False
        self._jumped_since_read = False
        self.last_jump_mps = 0.0
        self.last_jump_dps = 0.0

    def write(self, pos: tuple | None, epoch: int) -> None:
        with self._lock:
            if pos is None:
                # Latch it: the control loop must see that tracking was lost
                # even if it happened between two of its own reads.
                self._lost_since_read = True
                return
            new_pos = np.asarray(pos[:3], dtype=float)
            # track_stereo returns (x,y,z, qx,qy,qz,qw) -- see
            # OrbSlamTracker.track_stereo. scipy takes quaternions in
            # exactly that (x,y,z,w) order, so no reordering here.
            new_rot = Rotation.from_quat(np.asarray(pos[3:7], dtype=float))
            now = time.monotonic()
            if self._pos is not None and now > self._t:
                dt = now - self._t
                speed = float(np.linalg.norm(new_pos - self._pos) / dt)
                spin = float(np.degrees((new_rot * self._rot.inv()).magnitude()) / dt)
                if speed > POSE_JUMP_MPS or spin > POSE_JUMP_DPS:
                    # Latched for the same reason `lost` is: this must not be
                    # missed just because it happened between two control
                    # cycles. Note the new pose is still stored -- the point
                    # is to disengage, not to hide where tracking now thinks
                    # it is.
                    self._jumped_since_read = True
                    self.last_jump_mps = speed
                    self.last_jump_dps = spin
            self._pos = new_pos
            self._rot = new_rot
            self._t = now
            self._epoch = epoch

    def read(self) -> tuple[np.ndarray | None, float, int | None, bool, bool]:
        with self._lock:
            lost, jumped = self._lost_since_read, self._jumped_since_read
            self._lost_since_read = self._jumped_since_read = False
            return ((None if self._pos is None else self._pos.copy()), self._rot,
                    self._t, self._epoch, lost, jumped)


class KeyClutch:
    """Non-blocking single-keypress reader for the engage toggle.

    Raw terminal mode so a keypress registers without Enter, restored on
    exit including on exception. Degrades to disabled (never engages) if
    stdin is not a tty, so piping or nohup can't silently leave the arm
    permanently engaged."""

    def __init__(self) -> None:
        self.enabled = sys.stdin.isatty()
        self._old: Any = None

    def __enter__(self) -> "KeyClutch":
        if self.enabled:
            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.enabled and self._old is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)

    def poll(self) -> str | None:
        if not self.enabled:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1).lower()
        return None


def parse_axis_map(spec: str) -> np.ndarray:
    """'+z,-x,-y' -> the 3x3 signed permutation taking camera axes to robot
    axes. See the module docstring on how to find the right one."""
    axes = {"x": 0, "y": 1, "z": 2}
    parts = [p.strip().lower() for p in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--axis-map needs three comma-separated terms, got {spec!r}")
    m = np.zeros((3, 3))
    for row, p in enumerate(parts):
        sign = -1.0 if p.startswith("-") else 1.0
        letter = p.lstrip("+-")
        if letter not in axes:
            raise ValueError(f"--axis-map term {p!r} is not one of +x,-x,+y,-y,+z,-z")
        m[row, axes[letter]] = sign
    return m


def fit_frame_rotation(cam_dirs: list) -> np.ndarray:
    """Least-squares rotation taking camera coordinates to robot coordinates,
    from the three directions captured during calibration.

    Why this exists alongside --axis-map: an axis permutation can only
    express multiples of 90 degrees, and the frame offset here is not one.
    ORB-SLAM3's world frame has its Z aligned to gravity by the IMU, but its
    YAW is set by wherever the camera happened to be pointing when the map
    initialized -- effectively arbitrary, and different every session unless
    a saved atlas fixes it. Snapping that to the nearest axis leaves up to
    45 degrees of yaw error. Translation survives that as a cosine factor;
    rotation does not, because the whole rotation AXIS ends up tilted, which
    is what makes the wrist turn about visibly the wrong axis.

    Kabsch/SVD, with the determinant forced to +1 so the result is a proper
    rotation rather than a reflection -- see the same check in the
    calibration output."""
    A = np.column_stack([np.asarray(d, float) / np.linalg.norm(d) for d in cam_dirs])
    W, _, Vt = np.linalg.svd(np.linalg.inv(A))
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(W @ Vt)))])
    return W @ D @ Vt


def dominant_axis_term(delta_cam: np.ndarray) -> tuple[str, float]:
    """Reduce a captured camera-frame motion to the single axis-map term it
    implies, plus how cleanly it separated.

    The ratio returned is |largest component| / |second largest|. Near 1.0
    means the motion was diagonal in camera coordinates and the answer is a
    coin flip -- the caller should make the operator repeat it rather than
    silently pick one. A clean single-axis push gives 3x or better."""
    mag = np.abs(delta_cam)
    order = np.argsort(mag)[::-1]
    best = int(order[0])
    ratio = float(mag[best] / mag[order[1]]) if mag[order[1]] > 1e-9 else float("inf")
    sign = "+" if delta_cam[best] >= 0 else "-"
    return f"{sign}{'xyz'[best]}", ratio


def run_axis_calibration(slot: "PoseSlot", keys: "KeyClutch", ui: "QuietStdout") -> int:
    """Interactive: ask for one robot axis at a time, watch which camera axis
    moved, print the resulting --axis-map.

    Everything is drawn as one small self-updating block rather than scrolling
    prompts, and the distance travelled is shown live while you move -- the
    operator's hands are on the gripper, not the keyboard, so "how far have I
    gone, is tracking still good" has to be readable at a glance without
    hunting back through output.

    Done as a guided capture rather than by hand because the failure mode of
    doing it by hand is silent: a sign error just means the arm runs the wrong
    way the first time it is allowed to move."""
    prompts = [
        ("X", "AWAY FROM YOU"),
        ("Y", "to your LEFT"),
        ("Z", "UPWARDS"),
    ]
    terms: list[str] = []
    captured: list[np.ndarray] = []
    ui.say("\n" + "=" * 58)
    ui.say("  AXIS CALIBRATION -- the arm is never touched in this mode")
    ui.say("=" * 58)
    ui.say("  For each of the 3 directions: press SPACE, move the gripper")
    ui.say("  at least 15cm that way, press SPACE again.   Q quits.")
    ui.say("=" * 58)

    def health(pos: Any, age: float) -> str:
        if pos is None:
            return "NO POSE YET   "
        if age > POSE_STALE_S:
            return "TRACKING LOST "
        return "tracking ok   "

    for step, (name, direction) in enumerate(prompts, start=1):
        while True:
            ui.say(f"\n  [{step}/3]  move the gripper  >>> {direction} <<<")
            ui.say(f"          (this becomes the robot's +{name} direction)")

            # --- phase 1: wait for SPACE, showing tracking health ---
            start_pos = None
            while start_pos is None:
                k = keys.poll()
                if k == "q":
                    ui.say("\n  aborted.")
                    return 1
                pos, _, t, _, _, _ = slot.read()
                age = time.monotonic() - t
                if k == " ":
                    if pos is None or age > POSE_STALE_S:
                        ui.say("\r          cannot start -- no tracking. Point the camera at "
                               "textured scenery.        ")
                        time.sleep(0.6)
                    else:
                        start_pos = pos
                        break
                ui.say(f"\r          {health(pos, age)}  press SPACE to start        ", end="")
                time.sleep(0.05)
            ui.say("")

            # --- phase 2: live distance while moving ---
            end_pos, failure = None, None
            while end_pos is None:
                k = keys.poll()
                if k == "q":
                    ui.say("\n  aborted.")
                    return 1
                pos, _, t, _, _, lost = slot.read()
                age = time.monotonic() - t
                if lost or pos is None or age > POSE_STALE_S:
                    failure = "tracking was lost during the move"
                    break
                travel = float(np.linalg.norm(pos - start_pos)) * 100.0
                far_enough = "OK, press SPACE" if travel >= 15.0 else "keep going     "
                ui.say(f"\r          moved {travel:5.1f} cm   {far_enough}        ", end="")
                if k == " ":
                    end_pos = pos
                time.sleep(0.05)
            ui.say("")

            if failure is None:
                delta = end_pos - start_pos
                travel = float(np.linalg.norm(delta))
                if travel < 0.05:
                    failure = f"only {travel*100:.0f}cm of motion -- too little to read a direction from"
                else:
                    term, ratio = dominant_axis_term(delta)
                    if ratio < 2.0:
                        failure = (f"that move was diagonal in camera coordinates "
                                   f"(separation {ratio:.1f}x) -- move along one direction only")

            if failure is not None:
                ui.say(f"          RETRY: {failure}")
                continue

            ui.say(f"          captured {travel*100:.0f}cm  ->  robot {name} follows camera "
                   f"{term}  ({ratio:.0f}x clean)")
            terms.append(term)
            captured.append(delta)
            break

    spec = ",".join(terms)
    ui.say("\n" + "=" * 58)
    if len({t.lstrip("+-") for t in terms}) != 3:
        ui.say(f"  '{spec}' reuses a camera axis, so it is not a valid mapping.")
        ui.say("  Two of the three moves went the same physical way. Redo it.")
        ui.say("=" * 58)
        return 1
    # A rigid camera mounting can only ever produce a proper rotation. A
    # determinant of -1 is a mirrored frame, which no physical mounting can
    # give -- so it means exactly one of the three directions was captured
    # with the wrong sign. Worth checking because that is the error whose
    # only other symptom is the arm running the wrong way under --execute.
    det = float(np.linalg.det(parse_axis_map(spec)))
    if det < 0:
        ui.say(f"  '{spec}' is a MIRRORED frame (determinant {det:+.0f}), which no physical camera")
        ui.say("  mounting can produce. Exactly one of the three moves went the opposite way")
        ui.say("  to what was asked. Redo it, watching the direction names carefully.")
        ui.say("=" * 58)
        return 1
    ui.say(f'  RESULT:   --axis-map "{spec}"')

    # The permutation above is the readable summary; this is the number the
    # script should actually use, especially with --rotation.
    R = fit_frame_rotation(captured)
    residual = float(np.degrees(np.arccos(np.clip(
        (np.trace(parse_axis_map(spec).T @ R) - 1) / 2, -1, 1))))
    out = Path(FRAME_FILE)
    out.write_text(json.dumps({
        "_comment": ("Camera-frame -> robot-base rotation, fitted from the three calibration "
                     "moves. Preferred over the axis-map string, which can only express 90-degree "
                     "steps; see fit_frame_rotation() for why that matters for --rotation. "
                     "Re-run --calibrate-axes after remounting the camera, or after any session "
                     "whose map initialized from a different starting orientation (ORB-SLAM3's "
                     "world yaw is arbitrary unless a saved atlas fixes it)."),
        "axis_map": spec,
        "rotation": R.tolist(),
        "offset_from_axis_map_deg": round(residual, 2),
    }, indent=1))
    ui.say(f"            saved the full fitted frame to {out}")
    ui.say(f"            it differs from the axis-map by {residual:.1f} deg of yaw --")
    if residual > 5.0:
        ui.say(f"            enough to make --rotation turn about a visibly wrong axis,")
        ui.say(f"            so pass --frame {out} instead of --axis-map.")
    else:
        ui.say(f"            small, so either will behave much the same.")
    ui.say("=" * 58)
    ui.say("  Next: re-run in dry run and check the printed arm deltas move the")
    ui.say("  way you expect, before adding --execute.\n")
    return 0


def clamp_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n * max_norm if n > max_norm else v


def setup_arm(ip: str, execute: bool, ui: "QuietStdout") -> Any:
    """Connect and put the arm in servo mode. Returns None in dry run, so
    every later arm call has to be guarded and a dry run genuinely cannot
    command anything -- rather than relying on an `if execute` around each
    call site being remembered."""
    if not execute:
        ui.say(f"[arm] DRY RUN -- not connecting to {ip}. Targets will be printed only.")
        return None
    from xarm.wrapper import XArmAPI  # imported late: dry run must not need the SDK

    ui.say(f"[arm] connecting to {ip}")
    arm = XArmAPI(ip)
    arm.motion_enable(enable=True)
    arm.clean_error()
    arm.clean_warn()
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)
    code, pose = arm.get_position()
    if code != 0:
        raise RuntimeError(f"xArm get_position failed: code={code}")
    ui.say(f"[arm] current pose: {np.round(pose, 2).tolist()}")
    arm.set_mode(1)   # servo motion mode
    arm.set_state(0)
    time.sleep(0.5)
    return arm


# xArm controller error codes worth naming at the point of failure, rather
# than leaving the operator to look up a bare number mid-session. Full table:
# xArm-Python-SDK/doc/api/xarm_api_code.md
CONTROLLER_ERRORS = {
    21: "Kinematic error -- no IK solution for that target",
    22: "Self-collision",
    23: "Joint angle exceeded its limit",
    24: "Speed exceeded limit",
    25: "Planning error",
    31: "Collision (abnormal current)",
    35: "Safety boundary limit",
}


def explain_servo_rejection(arm: Any, ret: int, ui: "QuietStdout") -> None:
    """Say what the controller actually objected to, then clear it.

    Both halves matter. A bare `returned 1` sends you to a code table
    mid-session; and once error_code is non-zero the controller refuses
    every subsequent command, so without clearing it here one rejection
    ends the session even though re-anchoring would have been enough.
    The specific-detail queries must run BEFORE clean_error() -- they read
    the very state it discards."""
    code = arm.error_code
    ui.say(f"\n[arm] command rejected (ret={ret}), controller error {code}: "
           f"{CONTROLLER_ERRORS.get(code, 'see doc/api/xarm_api_code.md')}")
    if code == 23:
        try:
            ok, info = arm.get_c23_error_info()
            if ok == 0 and info:
                for servo_id, angle in info:
                    ui.say(f"       joint J{servo_id} is at {angle:.1f} deg, past its limit")
        except Exception:
            pass
    if code == 24:
        try:
            ok, info = arm.get_c24_error_info()
            if ok == 0 and info:
                ui.say(f"       joint speed detail: {info}")
        except Exception:
            pass
    ui.say("       clearing the error so you can re-anchor; move the arm back toward the middle")
    ui.say("       of its range first, or the same target will be rejected again.")
    try:
        arm.clean_error()
        arm.clean_warn()
        arm.set_mode(1)
        arm.set_state(0)
    except Exception as exc:  # noqa: BLE001
        ui.say(f"       could not clear it automatically ({exc}) -- "
               f"see ~/projects/teleop_collection/scripts/hard_recover.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true",
                        help="Actually command the arm. Without this the script runs fully but sends "
                             "nothing, printing the targets it would have sent (default: dry run).")
    parser.add_argument("--calibrate-axes", action="store_true",
                        help="Guided capture to find --axis-map: move the gripper along each robot "
                             "axis in turn and this prints the mapping. Never touches the arm.")
    parser.add_argument("--ip", default=DEFAULT_XARM_IP, help=f"xArm IP. Default: {DEFAULT_XARM_IP}")
    parser.add_argument("--axis-map", default="+x,+y,+z",
                        help="Camera axes -> robot axes, e.g. '+z,-x,-y'. The default identity is "
                             "almost certainly wrong for a real mounting -- see the module docstring "
                             "for how to find yours from a dry run.")
    parser.add_argument("--scale", type=float, default=SCALE_MM_PER_M,
                        help=f"Millimetres of arm motion per metre of hand motion. Default "
                             f"{SCALE_MM_PER_M:.0f} (the value tuned for the Quest teleop); 1000 is 1:1.")
    parser.add_argument("--max-delta-mm", type=float, default=MAX_CART_DELTA_MM,
                        help=f"Hard cap on distance from the anchor. Default {MAX_CART_DELTA_MM:.0f}mm.")
    parser.add_argument("--rotation", action="store_true",
                        help="Also drive the arm's wrist orientation from the gripper's, not just "
                             "position (default: off). Needs the same --axis-map. Off by default "
                             "because tracking noise reaches orientation directly and a degree of "
                             "wrist error is far more visible at the fingertips than a millimetre "
                             "of position error -- get position working first.")
    parser.add_argument("--frame", type=Path, default=None,
                        help=f"Use the full fitted camera->robot rotation written by "
                             f"--calibrate-axes (default file: {FRAME_FILE}) instead of "
                             f"--axis-map. Preferred with --rotation: an axis map can only "
                             f"express 90-degree steps, and ORB-SLAM3's world yaw is arbitrary.")
    parser.add_argument("--rot-scale", type=float, default=ROT_SCALE,
                        help=f"Degrees of arm rotation per degree of hand rotation. Default "
                             f"{ROT_SCALE} (turn your hand twice as far as the wrist should go); "
                             f"1.0 is 1:1. Lower this first if rotation feels twitchy.")
    parser.add_argument("--rot-alpha", type=float, default=ROT_ALPHA,
                        help=f"Rotation smoothing, separate from --alpha. Lower is calmer and "
                             f"laggier. Default {ROT_ALPHA}.")
    parser.add_argument("--max-rot-deg", type=float, default=MAX_ROT_DEG,
                        help=f"With --rotation, cap on angle away from the anchor orientation. "
                             f"Default {MAX_ROT_DEG:.0f} deg.")
    parser.add_argument("--max-step-mm", type=float, default=MAX_STEP_MM,
                        help=f"Cap on how far the commanded pose may move per 20ms control cycle. "
                             f"Default {MAX_STEP_MM:.0f}mm (~0.4 m/s). Bounds arm speed whatever the "
                             f"pose stream does.")
    parser.add_argument("--alpha", type=float, default=TARGET_ALPHA,
                        help=f"Smoothing: higher follows the hand more closely, lower is calmer. "
                             f"Default {TARGET_ALPHA}.")
    parser.add_argument("--orbslam-map-dir", type=Path, default=None,
                        help="Pre-built atlas dir to localize against (see build_atlas_map.py). "
                             "Strongly preferred for teleop: cold-starting a map means resets, and "
                             "every reset disengages the clutch.")
    parser.add_argument("--orbslam-settings", type=Path, default=None,
                        help="Override the settings YAML. Defaults to the relocalize variant when "
                             "--orbslam-map-dir is given, otherwise the plain one.")
    parser.add_argument("--orbslam-vocab", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--orbslam-lib", type=Path, default=DEFAULT_LIB_PATH)
    parser.add_argument("--verbose", action="store_true",
                        help="Let ORB-SLAM3's console output through instead of routing it to a log "
                             "file. Use it when tracking misbehaves and you want to see why live.")
    parser.add_argument("--viewer", action="store_true",
                        help="Show ORB-SLAM3's Pangolin viewer (default: off -- the terminal status "
                             "line is the thing to watch while teleoperating).")
    args = parser.parse_args()

    if args.frame is not None:
        spec = json.loads(Path(args.frame).read_text())
        axis_map = np.asarray(spec["rotation"], dtype=float)
        print(f"[setup] frame from {args.frame}: axis-map '{spec.get('axis_map','?')}' plus "
              f"{spec.get('offset_from_axis_map_deg', 0)} deg of yaw the permutation cannot express")
    else:
        axis_map = parse_axis_map(args.axis_map)
    if args.frame is None and args.axis_map.replace(" ", "") == "+x,+y,+z" and not args.calibrate_axes:
        print("[warn] --axis-map is the identity. Unless the camera frame happens to align with the "
              "arm's base frame, motion will go the wrong way. Find yours in a dry run first.")

    settings = args.orbslam_settings or (
        ORBSLAM_DEFAULT_RELOCALIZE_SETTINGS_PATH if args.orbslam_map_dir is not None
        else ORBSLAM_DEFAULT_SETTINGS_PATH
    )
    if args.orbslam_map_dir is None:
        print("[warn] no --orbslam-map-dir: this cold-starts a map, which is exactly the case that "
              "reset-loops during careful motion. Expect frequent disengages.")

    import pyrealsense2 as rs

    # ORB-SLAM3 is loud on stdout from C++ and it drowns out this script's own
    # prompts. Quiet by default; --verbose keeps everything inline.
    log_path = Path("umi_teleop_orbslam.log")
    quiet = QuietStdout(log_path, enabled=not args.verbose)

    capture = cam.RealSenseCapture(
        rs=rs, scan_dir=Path("."), max_duration_seconds=None,
        requested_color=None, requested_depth=None, enable_imu=True,
        record_rgb=False, record_depth=False, record_ir=True,
        # record_ir gates the IR stream AND the hooks this needs, but it
        # also gates saving -- teleoperation is live control, so nothing
        # should be left on disk wherever the operator happened to launch
        # from. Recording is synced_capture.py's job.
        save_to_disk=False,
        queue_size=256, imu_fps=200, show_preview=False, preview_fps=15.0,
    )
    quiet.__enter__()
    if quiet.enabled:
        quiet.say(f"[setup] ORB-SLAM3 console output -> {log_path}  (--verbose to see it inline)")
    quiet.say("[setup] starting camera and tracker, this takes a moment ...")
    capture.start_camera()

    # ORB-SLAM3 hardcodes the atlas load path relative to the process cwd at
    # construction time -- same chdir dance as construct_orb_tracker().
    original_cwd = Path.cwd()
    try:
        if args.orbslam_map_dir is not None:
            os.chdir(args.orbslam_map_dir)
        tracker = OrbSlamTracker(settings_path=settings.resolve(), vocab_path=args.orbslam_vocab.resolve(),
                                 lib_path=args.orbslam_lib.resolve(), use_viewer=args.viewer)
    finally:
        os.chdir(original_cwd)

    slot = PoseSlot()
    worker = OrbSlamWorker(tracker, lambda ts, pose, epoch: slot.write(pose, epoch),
                           gripper_masks=load_stereo_masks())

    last_gyro = [0.0, 0.0, 0.0]
    have_gyro = [False]

    def _on_imu_sample(kind: str, row: dict[str, Any]) -> None:
        if kind == "gyro":
            last_gyro[:] = [row["x"], row["y"], row["z"]]
            have_gyro[0] = True
        elif kind == "accel" and have_gyro[0]:
            tracker.feed_imu(row["host_time_ns"] / 1e9, *last_gyro, row["x"], row["y"], row["z"])

    def _on_stereo_frame(frame_id: int, host_time_ns: int, left: Any, right: Any) -> None:
        worker.submit(capture.frame_rows[-1]["host_time_ns"] / 1e9, left, right)

    capture.on_imu_sample = _on_imu_sample
    capture.on_stereo_frame = _on_stereo_frame

    arm = setup_arm(args.ip, args.execute and not args.calibrate_axes, quiet)

    if not args.calibrate_axes:
        quiet.say("\n  SPACE  engage / disengage        R  re-anchor        Q  quit")
        quiet.say("  Tracking loss, a map reset or a pose jump disengages automatically.\n")

    # Camera capture runs on its own thread so this one can be the control
    # loop; capture.capture() blocks until stop_event.
    threading.Thread(target=capture.capture, daemon=True).start()

    engaged = False
    cam_anchor: np.ndarray | None = None
    cam_rot_anchor: Any = None
    arm_rot_anchor: Any = None
    arm_anchor: np.ndarray | None = None
    rot_smooth: Any = Rotation.identity()
    epoch_anchor: int | None = None
    delta_smooth = np.zeros(3)
    rate_limited = 0
    last_status = 0.0
    rc = 0

    def disengage(why: str) -> None:
        nonlocal engaged, cam_anchor, arm_anchor, epoch_anchor, delta_smooth, rate_limited
        nonlocal cam_rot_anchor, arm_rot_anchor, rot_smooth
        if engaged:
            quiet.say(f"\n[clutch] DISENGAGED -- {why}. Press SPACE to re-anchor and continue.")
        engaged = False
        cam_anchor = arm_anchor = epoch_anchor = None
        delta_smooth = np.zeros(3)
        cam_rot_anchor = arm_rot_anchor = None
        rot_smooth = Rotation.identity()
        rate_limited = 0

    try:
        with KeyClutch() as keys:
            if not keys.enabled:
                quiet.say("[warn] stdin is not a terminal -- the clutch can never engage. Run "
                          "this interactively.")
            if args.calibrate_axes:
                rc = run_axis_calibration(slot, keys, quiet)
                raise _CalibrationDone
            while True:
                loop_start = time.monotonic()
                pos, rot, pose_t, epoch, lost_latched, jumped = slot.read()

                key = keys.poll()
                if key == "q":
                    quiet.say("\n[quit]")
                    break
                if key == " ":
                    if engaged:
                        disengage("space pressed")
                    elif pos is None:
                        quiet.say("\n[clutch] cannot engage: no tracked pose yet.")
                    else:
                        engaged = True
                        cam_anchor = None  # forces a fresh anchor below
                if key == "r" and engaged:
                    cam_anchor = None
                    quiet.say("\n[clutch] re-anchoring")

                # --- every path that makes the pose stream discontinuous ---
                if engaged:
                    if lost_latched or pos is None:
                        disengage("tracking lost")
                    elif time.monotonic() - pose_t > POSE_STALE_S:
                        disengage(f"no pose for >{POSE_STALE_S:.1f}s")
                    elif epoch_anchor is not None and epoch != epoch_anchor:
                        disengage(f"map reset (epoch {epoch_anchor} -> {epoch})")
                    elif jumped:
                        disengage(f"pose jumped at {slot.last_jump_mps:.1f} m/s -- faster than a hand "
                                  f"can move, so the pose stream is discontinuous, not the gripper")

                if engaged and cam_anchor is None:
                    if arm is not None:
                        code, cur = arm.get_position()
                        if code != 0:
                            disengage(f"could not read arm pose (code {code})")
                            continue
                        arm_anchor = np.asarray(cur, dtype=float)
                    else:
                        arm_anchor = np.zeros(6)
                    cam_anchor = pos.copy()
                    cam_rot_anchor = rot
                    arm_rot_anchor = Rotation.from_euler(
                        XARM_EULER_SEQ, arm_anchor[3:6], degrees=True)
                    epoch_anchor = epoch
                    delta_smooth = np.zeros(3)
                    rot_smooth = Rotation.identity()
                    quiet.say(f"\n[clutch] ENGAGED. camera anchor {np.round(cam_anchor, 3).tolist()}  "
                          f"arm anchor {np.round(arm_anchor[:3], 1).tolist()}")

                if engaged:
                    delta_cam = pos - cam_anchor
                    if np.linalg.norm(delta_cam) < DEADZONE_M:
                        delta_cam = np.zeros(3)
                    delta_raw = clamp_norm(args.scale * (axis_map @ delta_cam), args.max_delta_mm)
                    stepped = (1.0 - args.alpha) * delta_smooth + args.alpha * delta_raw
                    # Second guard (see MAX_STEP_MM): bound the commanded
                    # step itself, so no single cycle can produce a fast
                    # move regardless of what upstream did. Applied AFTER
                    # smoothing because smoothing is what turns a target
                    # jump into a large single step in the first place.
                    step = stepped - delta_smooth
                    step_mm = float(np.linalg.norm(step))
                    if step_mm > args.max_step_mm:
                        step = step / step_mm * args.max_step_mm
                        rate_limited += 1
                    else:
                        rate_limited = 0
                    delta_smooth = delta_smooth + step
                    # Sustained clamping means the target is running away
                    # rather than one sample being noisy -- a slow drag
                    # toward a wrong pose is exactly what the per-cycle cap
                    # alone would quietly allow.
                    if rate_limited > RATE_LIMIT_PATIENCE:
                        disengage(f"target outran the rate limit for {RATE_LIMIT_PATIENCE} cycles")
                        continue

                    target = arm_anchor.copy()
                    target[0:3] = arm_anchor[0:3] + delta_smooth

                    if args.rotation and rot is not None and cam_rot_anchor is not None:
                        # How the hand turned, in the world frame the camera
                        # poses live in. World frame (new * old^-1), not body
                        # frame, so that turning the hand turns the tool the
                        # same way as seen from the room -- which is what an
                        # operator watching the arm expects.
                        d_cam = rot * cam_rot_anchor.inv()
                        # Re-express that rotation in the arm's base frame.
                        # A rotation changes basis by conjugation, M R M^T --
                        # the same axis_map that permutes translation axes,
                        # applied to both sides.
                        d_robot = Rotation.from_matrix(axis_map @ d_cam.as_matrix() @ axis_map.T)
                        rotvec = d_robot.as_rotvec(degrees=True)
                        angle = float(np.linalg.norm(rotvec))
                        # Soft deadzone: subtract it rather than gate on it,
                        # so there is no step at the threshold.
                        if angle <= ROT_DEADZONE_DEG:
                            rotvec = np.zeros(3)
                        else:
                            rotvec = rotvec / angle * (angle - ROT_DEADZONE_DEG)
                        rotvec *= args.rot_scale
                        angle = float(np.linalg.norm(rotvec))
                        if angle > args.max_rot_deg:
                            rotvec = rotvec / angle * args.max_rot_deg
                        # Smooth and rate-limit in rotation-vector space, so
                        # the same EMA and per-cycle cap as translation apply
                        # without any Euler-angle wrapping to get wrong.
                        prev = rot_smooth.as_rotvec(degrees=True)
                        stepped_rv = (1.0 - args.rot_alpha) * prev + args.rot_alpha * rotvec
                        rstep = stepped_rv - prev
                        rstep_deg = float(np.linalg.norm(rstep))
                        if rstep_deg > MAX_ROT_STEP_DEG:
                            rstep = rstep / rstep_deg * MAX_ROT_STEP_DEG
                        rot_smooth = Rotation.from_rotvec(prev + rstep, degrees=True)
                        target[3:6] = (rot_smooth * arm_rot_anchor).as_euler(
                            XARM_EULER_SEQ, degrees=True)

                    if arm is not None:
                        if arm.state == 4:
                            quiet.say("\n[arm] entered STOP state 4 -- aborting.")
                            rc = 1
                            break
                        ret = arm.set_servo_cartesian([float(v) for v in target], speed=SPEED,
                                                      mvacc=MVACC, is_radian=False)
                        if ret != 0:
                            explain_servo_rejection(arm, ret, quiet)
                            disengage("servo command rejected")

                    if loop_start - last_status > STATUS_DT:
                        last_status = loop_start
                        tag = "SERVO" if arm is not None else "DRY  "
                        quiet.say(f"\r[{tag}] cam d={np.round(delta_cam, 3).tolist()}  "
                              f"-> arm d={np.round(delta_smooth, 1).tolist()} mm  epoch={epoch}   ",
                              end="")
                elif loop_start - last_status > STATUS_DT:
                    last_status = loop_start
                    state = "no pose" if pos is None else "idle -- press SPACE"
                    quiet.say(f"\r[----- ] {state}   epoch={epoch}          ", end="", flush=True)

                time.sleep(max(0.0, LOOP_DT - (time.monotonic() - loop_start)))
    except _CalibrationDone:
        pass
    except KeyboardInterrupt:
        quiet.say("\n[quit] interrupted")
    finally:
        capture.stop_event.set()
        worker.close()
        capture.stop_camera()
        quiet.__exit__(None, None, None)
        tracker.shutdown()
        tracker.close()
        if arm is not None:
            # Back out of servo mode before releasing the arm, so it isn't
            # left waiting on a servo stream that stopped.
            arm.set_mode(0)
            arm.set_state(0)
            arm.disconnect()
            print("[arm] returned to mode 0 and disconnected.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
