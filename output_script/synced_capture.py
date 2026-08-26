#!/usr/bin/env python3
"""Record the RealSense camera (RGB + IMU) and the XL330 servo's position at
the same time, with a live monitor, and write out one synced dataset.

This reuses the two existing pieces as-is instead of reimplementing them:
  - camera/camera_collecting.py's RealSenseCapture drives the RealSense
    pipeline exactly like the standalone script does (same frame saving,
    same gyro/accel-to-frame matching).
  - servo/spring_position_mode.py's DynamixelPositionSpring configures the
    servo as a compliant "spring" (Current-based Position Control Mode,
    onboard PID) exactly like the standalone spring script does, so the
    servo still behaves like a spring -- pushing back toward center, safe to
    hold/move by hand -- WHILE its position is being logged. That's the
    point of recording through this script rather than reading a rigid,
    un-driven encoder: you can demonstrate a motion by hand against a known,
    repeatable restoring force and have it logged frame-by-frame.

## How "sync" actually works here

There's no hardware trigger tying the camera and the servo together -- they
are two independent USB devices with two independent clocks. What ties them
together is a single shared clock: the HOST's wall clock (time.time_ns()).

  - Every camera frame is already timestamped with host_time_ns the moment
    it arrives on this machine (see camera_collecting.py's _frame_callback).
  - Every servo position read is timestamped with host_time_ns the moment
    the read call returns, by the ServoPoller thread below.
  - At the end of the run, each frame gets matched to the nearest-in-time
    servo sample (same nearest-timestamp technique camera_collecting.py
    already uses to match IMU samples to frames -- see
    match_samples_to_frames there / match_servo_to_frames below), and the
    per-frame sync error (in ms) is written out alongside it so you can see
    how good the match actually was, not just assume it.

The camera runs at 60fps by default (see camera/camera_collecting.py's
COLOR_PROFILES). The servo's own USB link cannot reliably be read at a full
60Hz -- see servo/README.md's "Why not just compute the spring force from
the PC?" section: a combined position+velocity+current read measured out to
only ~31Hz on this hardware, because of ~16ms of USB/FTDI latency PER
transaction (round trip, so ~32ms). ServoPoller therefore polls in a tight
loop with no artificial delay (the transaction latency paces it on its own)
and just timestamps whatever rate it actually achieves -- typically
somewhere well under 60Hz. The frame-matching step above is what turns that
into a "one row per camera frame" synced table regardless of the servo's own
poll rate; the trade-off is that between two servo reads, several camera
frames may end up matched to the same (slightly stale) servo sample. A sync
summary is printed at the end of every run, telling you exactly how stale,
on average and worst-case, for that specific run.

## Live monitor

ONE cv2 window (see CombinedMonitor) shows the latest camera frame with a
rolling servo-position plot composited alongside it, side by side, both
updated while recording runs. This used to be two separate windows (a cv2
camera preview + a matplotlib plot) -- matplotlib's Tk backend on this
machine was too slow to redraw live video without visible lag, no matter
what thread it ran on, so the plot is now drawn with plain OpenCV drawing
calls (polylines + putText) instead, which are cheap enough to redraw at
video rate without becoming the bottleneck themselves. That's also what
lets both live in the same window: it's all just one image, composited and
shown with a single cv2.imshow() call.

Redraw rate is decoupled from the recording rate (--monitor-fps, default
15Hz) -- the same reason camera_collecting.py decouples --preview-fps from
--color-fps. Press 'q' or close the window to stop recording early, same as
camera_collecting.py's own --preview.

Usage:
    python3 synced_capture.py
    python3 synced_capture.py --max-current 100 --p-gain 80 --monitor-fps 10
    python3 synced_capture.py --no-monitor --max-duration 30
    python3 synced_capture.py --episodes 100
        Batch mode -- e.g. collecting N demonstrations for imitation
        learning back to back. Camera connection and servo torque stay up
        for the whole session (only each episode's own state resets, see
        RealSenseCapture.reset_for_new_episode), so episode 2+ starts
        instantly instead of reconnecting. Each episode still needs its own
        Enter/Space to start and stop, in terminal or monitor window, since
        a human performs a variable-length demonstration each time. Press
        q/Esc or close the monitor window at any point to stop the whole
        batch early (a plain Enter/Space only ends the current episode).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "camera"))
sys.path.insert(0, str(REPO_ROOT / "servo"))

import camera_collecting as cam  # noqa: E402
import spring_position_mode as servo_mod  # noqa: E402
from openvins_bridge import DEFAULT_LIB_PATH as OPENVINS_DEFAULT_LIB_PATH, OpenVinsTracker  # noqa: E402
from orbslam_bridge import (  # noqa: E402
    DEFAULT_LIB_PATH as ORBSLAM_DEFAULT_LIB_PATH,
    DEFAULT_SETTINGS_PATH as ORBSLAM_DEFAULT_SETTINGS_PATH,
    DEFAULT_VOCAB_PATH as ORBSLAM_DEFAULT_VOCAB_PATH,
    OrbSlamTracker,
)

# Default OpenVINS estimator config for this rig's D435I -- see
# open_vins/config/rs_d435i_umi/. Always overridable with --openvins-config.
# OpenVINS (monocular) is now OFF by default (see --openvins) -- kept
# available for side-by-side comparison/debugging, but ORB-SLAM3 (see
# --orbslam) is the primary live tracker: empirical testing this project's
# history showed OpenVINS drifts unboundedly under continuous handheld
# motion with no pauses, a failure mode ORB-SLAM3's Stereo-Inertial mode
# (real metric scale from the known IR baseline, not estimated online)
# doesn't share.
DEFAULT_OPENVINS_CONFIG = Path("/home/hakan/Desktop/umi/open_vins/config/rs_d435i_umi/estimator_config.yaml")

# Default calibration table -- mapping_csv/mapping_function.csv, a 52-point
# kinematic mapping (gear_displacement in deg -> gripper_displacement in mm)
# derived from this gripper's rack-and-pinion geometry. See
# mapping_csv/mapping_function.png for the plotted curve and
# mapping_csv/gear_displacement.csv / gripper_displacement.csv /
# gear_rack_displacement.csv for the underlying time-series it was built
# from. Always overridable with --gripper-calibration-csv. Defined here
# (before CombinedMonitor/ServoPoller) rather than down by
# load_gripper_calibration because CombinedMonitor.__init__ references
# GRIPPER_MAX_WIDTH_MM as a parameter default, which is evaluated at class-
# definition time -- it has to already exist by the time that class is
# defined below, not just by the time it's first called.
DEFAULT_GRIPPER_CALIBRATION_CSV = REPO_ROOT / "mapping_csv" / "mapping_function.csv"

# Bias (deg) between the calibration CSV's own zero point
# (gear_displacement=0, the fully-closed end) and the servo's raw absolute
# reading at that same physical end stop. gear_displacement=0 IS the
# fully-closed position (gripper_displacement there is ~0mm).
#
# Re-measured 2026-08-26 after the gripper was physically reassembled.
# Starting point was servo/spring_position_mode.py's newly re-measured
# GRIPPER_CLOSED_DEG (-5.81, from a --measure-freeplay full-range test),
# same as the raw closed-end reading -- but, same as the prior assembly's
# calibration (221.0 vs. that assembly's GRIPPER_CLOSED_DEG of 216.9),
# that needed a further fine-tune beyond the raw closed-end reading:
# checking the live width readout (servo/check_gripper_width.py) against a
# physical ruler at the middle of the gripper's range showed the raw
# closed-end value alone (-5.81) read ~5mm too wide there. The curve's
# local slope near mid-range is ~0.83 mm/deg (computed directly from
# mapping_csv/mapping_function.csv), so a 5mm correction needed +6.05 deg
# -- confirmed against the ruler at +6.05, giving this value. Re-check the
# same way (check_gripper_width.py against a ruler at a few known
# openings, not just fully-closed) if this ever drifts again.
GRIPPER_CALIBRATION_BIAS_DEG = 0.24

# Real measured maximum gripper opening (mm). The CSV's kinematic model
# overshoots this near full-open (predicts up to ~85.4mm there) -- the
# idealized linkage geometry it was derived from doesn't capture whatever
# actually stops the gripper at 80mm in reality. Rather than trust the
# model past the real physical limit, interpolated widths are capped here;
# the CSV/interpolation itself is left untouched (confirmed by testing on
# 2026-08-19).
GRIPPER_MAX_WIDTH_MM = 80.0


class ServoPoller:
    """Background thread: reads Present Position/Velocity/Current from the
    servo as fast as the USB link allows (no artificial delay -- the ~16ms
    per-transaction FTDI latency paces this on its own, see module
    docstring), timestamping every successful read with the host clock.

    self.latest is updated on every read regardless of recording state, so
    the live monitor can show motion as soon as the spring is configured.
    self.samples only accumulates while `active_event` is set, i.e. only
    during the actual recording window -- mirrors RealSenseCapture.active,
    which gates camera frames the same way.
    """

    def __init__(self, servo: "servo_mod.DynamixelPositionSpring", active_event: threading.Event):
        self.servo = servo
        self.active_event = active_event
        self.samples: list[dict[str, Any]] = []
        self.latest: dict[str, Any] | None = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.error: Exception | None = None
        self.achieved_hz = 0.0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # A transient comm hiccup (plausible given the poller deliberately polls
    # with zero delay -- see class docstring) must not be allowed to freeze
    # `latest` forever at whatever value it last held: that's exactly what
    # produces a live plot that looks alive but is secretly stuck on a
    # constant value. Only give up after this many READS IN A ROW fail.
    MAX_CONSECUTIVE_ERRORS = 20

    def _run(self) -> None:
        window_start = time.monotonic()
        window_count = 0
        last_status_print = time.monotonic()
        reads_attempted = 0
        consecutive_errors = 0
        while not self.stop_event.is_set():
            reads_attempted += 1
            try:
                current_ma, velocity, angle = self.servo.read_state()
            except Exception as exc:  # noqa: BLE001 - surface, don't crash the thread silently
                consecutive_errors += 1
                self.error = exc
                if consecutive_errors == 1:
                    # Only the first failure in a burst gets printed --
                    # otherwise a real sustained failure floods the console.
                    print(f"\nServoPoller: read failed ({exc}); retrying...")
                if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                    print(f"\nServoPoller: giving up after {consecutive_errors} consecutive "
                          f"failures. Last error: {exc}")
                    break
                continue
            if consecutive_errors:
                print(f"\nServoPoller: recovered after {consecutive_errors} failed read(s).")
            consecutive_errors = 0
            self.error = None
            host_time_ns = time.time_ns()
            sample = {
                "host_time_ns": host_time_ns,
                "position_deg": angle,
                "velocity_deg_s": velocity,
                "current_ma": current_ma,
            }
            with self.lock:
                self.latest = sample
                if self.active_event.is_set():
                    self.samples.append(sample)

            window_count += 1
            now = time.monotonic()
            if now - window_start >= 1.0:
                self.achieved_hz = window_count / (now - window_start)
                window_start = now
                window_count = 0

            # Printed to the console every ~2s regardless of whether the
            # live plot window is even open -- makes "is servo data flowing
            # at all" answerable from the terminal alone, no GUI needed.
            if now - last_status_print >= 2.0:
                last_status_print = now
                recording = "recording" if self.active_event.is_set() else "not recording yet"
                print(f"\n[servo] {reads_attempted} reads, {self.achieved_hz:.1f} Hz, "
                      f"last position {angle:.2f} deg ({recording})")


class OrbSlamWorker:
    """Runs ORB-SLAM3 tracking on its own dedicated thread, always on the
    MOST RECENTLY submitted stereo frame -- never a backlog, and drops
    older frames if it's still busy when a newer one arrives.

    Why this exists: per-frame tracking (feature extraction + local mapping
    + loop closing, all running concurrently inside ORB-SLAM3) can take
    longer than the camera's frame interval. RealSenseCapture's own
    on_stereo_frame hook fires once per SAVED frame, in strict FIFO order,
    from the same thread that drains its video_queue -- calling
    tracker.track_stereo() synchronously from there (an earlier version of
    this integration did exactly that) means that once tracking falls
    behind real time, the backlog only grows: every subsequent frame is
    tracked further and further behind when it was actually captured.
    That's not just slow -- it starves ORB-SLAM3's internal IMU queue (the
    frame's timestamp lags behind the live IMU stream that keeps arriving
    in real time, so by the time a backlogged frame is tracked, most queued
    IMU samples are already "in its future" and get skipped -- this is what
    produced the near-constant "Empty IMU measurements vector!!!" warnings
    in live testing) and, empirically, crashed ORB-SLAM3's IMU
    initialization step consistently around 10s into two separate live
    test runs. The stock stereo_inertial_realsense_D435i.cc example never
    hits this: it has no queue at all, it just overwrites its one pending
    frame slot and always tracks whatever's newest.

    This class reproduces that same "always latest, drop if behind" model
    for our live-feedback path specifically -- it does NOT affect what gets
    saved to disk (frames.csv/ir_left.txt/ir_right.txt still get every
    frame, from RealSenseCapture's own synchronous save path). Only the
    live ORB-SLAM3 trajectory (camera_trajectory.csv) can end up with fewer
    rows than there were saved frames, when tracking is running slower than
    the camera -- each row is still a real, live-tracked pose (or an
    honest is_lost=1), never a stale/backlogged one.

    IMPORTANT, learned the hard way: "always latest" alone is NOT enough.
    IMU keeps flowing into the tracker continuously in real time regardless
    of how slow tracking is (see _on_imu_sample -- it's independent of this
    worker). If a single track_stereo() call takes longer than the camera's
    frame interval (and per-call cost tends to grow further as the map/
    keyframe database grows, from concurrent local mapping + loop closing),
    then by the time this worker finishes one call and grabs the next
    "latest" pending frame, that frame's own timestamp is already stale
    relative to how much IMU has meanwhile accumulated -- there's almost no
    IMU left that's chronologically BEFORE it, which is what produced the
    near-constant "Empty IMU measurements vector!!!" in live testing, and
    ultimately crashed ORB-SLAM3's IMU initialization step. Confirmed via a
    live diagnostic print in orb_capi.cc: the gap between a tracked frame's
    timestamp and the newest IMU sample fed alongside it grew steadily
    (multiple seconds within a couple seconds of wall-clock testing) even
    after switching both sides to a single shared host clock, ruling out a
    RealSense clock-domain mismatch and confirming this is genuine,
    worsening processing lag. min_interval below (plus a lower
    ORBextractor.nFeatures in the settings YAML) is the mitigation: give
    each attempt a realistic time budget so tracking has a chance to
    actually keep pace with the camera, rather than perpetually processing
    something several seconds in the past.
    """

    def __init__(self, tracker: "OrbSlamTracker", on_pose, min_interval: float = 0.1):
        self.tracker = tracker
        self.on_pose = on_pose  # callback(timestamp, pose_tuple_or_None, map_epoch)
        # Caps how often this worker will even ATTEMPT a track_stereo() call
        # (default 10Hz) -- see the class docstring's "learned the hard way"
        # note. This is distinct from "drop if busy": that alone still lets
        # submissions get accepted as fast as the camera delivers them
        # (~15-30Hz), which is more often than ORB-SLAM3 can realistically
        # sustain per call on this hardware at the current feature count, so
        # the "latest" pending frame was still routinely already stale by
        # the time it got picked up. Throttling acceptance itself, not just
        # processing, is what actually bounds that staleness.
        #
        # DO NOT lower this again without a lot more caution than "the
        # whole-episode average call duration has headroom". A prior attempt
        # measured avg=11ms/p95=17ms/max=37ms over one full episode
        # (nFeatures=800) and, on that basis alone, tried 0.05 (20Hz) --
        # which promptly reproduced the original crash (near-constant "Empty
        # IMU measurements vector!!!" then a segfault, seconds into a fresh
        # episode). Root cause: MAP INITIALIZATION (the first several frames
        # of every fresh episode, before "New Map created" settles) is far
        # more expensive per-call than steady-state tracking -- the
        # accepted-frame gaps during that failed run were ~470ms apart, not
        # the ~50ms the throttle should have allowed, meaning that startup
        # spike alone was enough to fall behind and spiral, and the whole-
        # episode average had completely hidden it (it gets diluted by many
        # cheap steady-state calls afterward). 0.1 (10Hz) is the only value
        # that has actually survived a live run without crashing during
        # recording (proven across several episodes now) -- treat it as the
        # floor, not a conservative guess to be optimized away. If revisiting
        # this, look at stats_summary()'s max/p95 from just the first ~2s of
        # a fresh episode specifically, not the whole-episode average.
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._pending: tuple[float, Any, Any] | None = None
        self._last_submit_time = 0.0
        self._stop_event = threading.Event()
        # Timing instrumentation, added while investigating the ~80% tracking
        # rate: cheap, in-process, printed as ONE summary line at close() --
        # not a per-frame flood like the diagnostic already removed from
        # orb_capi.cc. See the min_interval comment above for what it found.
        self._call_durations: list[float] = []
        self._call_gaps: list[float] = []
        self._last_call_start: float | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, timestamp: float, left_gray: Any, right_gray: Any) -> None:
        """Cheap, non-blocking -- called from RealSenseCapture's own capture-
        draining thread on every stereo frame (see
        _on_stereo_frame_for_orbslam in main()). Replaces whatever was
        pending (superseding/dropping it if the worker thread hasn't picked
        it up yet) -- UNLESS min_interval hasn't elapsed since the last
        accepted submission, in which case this frame is ignored entirely
        (see class docstring)."""
        now = time.monotonic()
        with self._lock:
            if now - self._last_submit_time < self.min_interval:
                return
            self._last_submit_time = now
            self._pending = (timestamp, left_gray, right_gray)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                item = self._pending
                self._pending = None
            if item is None:
                time.sleep(0.005)
                continue
            timestamp, left_gray, right_gray = item
            call_start = time.monotonic()
            if self._last_call_start is not None:
                self._call_gaps.append(call_start - self._last_call_start)
            self._last_call_start = call_start
            pose = self.tracker.track_stereo(timestamp, left_gray, right_gray)
            self._call_durations.append(time.monotonic() - call_start)
            # last_map_epoch is updated by track_stereo() above regardless of
            # whether tracking succeeded -- read it right after, same call
            # ordering the C API guarantees atomicity for (see
            # OrbSlamTracker.last_map_epoch's docstring).
            self.on_pose(timestamp, pose, self.tracker.last_map_epoch)

    def stats_summary(self) -> str:
        """One-line timing summary -- see the instrumentation comment in
        __init__. Safe to call any time; reports "no frames processed" if
        called before the first track_stereo() call completes.

        Reports the first 10 calls' durations SEPARATELY from the
        whole-episode average -- see the min_interval comment in __init__
        for why: map initialization (the first several frames of a fresh
        episode) is far more expensive than steady-state tracking, and a
        whole-episode average dilutes that spike away completely. Look at
        this "startup" figure, not just the overall average, before ever
        considering lowering min_interval again."""
        if not self._call_durations:
            return "ORB-SLAM3 timing: no frames processed"
        startup = self._call_durations[:10]
        parts = [
            f"ORB-SLAM3 timing: startup (first {len(startup)} calls) max={max(startup)*1000:.0f}ms"
        ]
        d = sorted(self._call_durations)
        mean_d = sum(d) / len(d)
        p95_d = d[int(0.95 * (len(d) - 1))]
        parts.append(
            f"whole-episode ({len(d)} calls) avg={mean_d*1000:.0f}ms "
            f"p95={p95_d*1000:.0f}ms max={d[-1]*1000:.0f}ms"
        )
        if self._call_gaps:
            g = self._call_gaps
            parts.append(
                f"call-to-call gap avg={sum(g)/len(g)*1000:.0f}ms min={min(g)*1000:.0f}ms "
                f"(min_interval floor={self.min_interval*1000:.0f}ms)"
            )
        return "; ".join(parts)

    def close(self) -> None:
        """Stops the worker thread. Does NOT close the underlying tracker --
        caller closes that separately (same split as ServoPoller/servo).
        Must be called (and joined) BEFORE closing/destroying the tracker
        this worker holds a reference to, so no in-flight track_stereo call
        can run against an already-destroyed handle."""
        self._stop_event.set()
        self._thread.join(timeout=2.0)


class CombinedMonitor:
    """ONE cv2 window: the latest camera frame with a rolling servo-position
    plot composited alongside it, side by side, shown with a single
    cv2.imshow() call. The plot panel is drawn with plain OpenCV primitives
    (polylines + putText), not matplotlib -- matplotlib (this machine's Tk
    backend) was too slow to redraw live video without visible lag no matter
    which thread it ran on.

    All cv2 calls (window creation, imshow, waitKey, destroy) run on ONE
    dedicated background thread owned by this class -- NOT the capture
    thread. This matters: even throttled, compositing (image copy + panel
    render + text + polyline + hstack) is real work, and running it inline
    inside on_frame (as an earlier version of this script did) reintroduced
    the exact camera-delay problem this file has already been through once.
    Isolating it on its own thread means however expensive a redraw gets, it
    can never stall camera frame draining -- the same principle that used to
    apply to the matplotlib version, just with a plain thread instead of
    "must be the main thread" (cv2's Qt/xcb backend just needs ONE
    consistent thread, not necessarily the process's main one).

    set_pending() is the only method the capture thread calls, on every
    frame -- cheap, just stores references behind a lock.
    """

    WINDOW_NAME = "Synced capture -- Enter/Space: start/stop episode, q/close: stop batch"

    # Panel look: white background, blue line, black text -- easy to change
    # here if the palette should change again.
    BG_COLOR = (255, 255, 255)     # BGR: white
    LINE_COLOR = (180, 119, 31)    # BGR: blue
    TEXT_COLOR = (20, 20, 20)      # BGR: near-black
    GRID_COLOR = (215, 215, 215)   # BGR: light gray
    GRID_DIVISIONS = 4              # grid cells per axis

    # Recording-status indicator (border around the whole window + a text
    # label) -- red/"RECORDING" while capture.active is set, gray/"waiting
    # to start" otherwise. See recording_active in __init__.
    RECORDING_COLOR = (0, 0, 220)   # BGR: red
    IDLE_COLOR = (140, 140, 140)    # BGR: gray
    STATUS_BORDER_PX = 8

    def __init__(
        self,
        cv2_module: Any,
        window_seconds: float,
        monitor_fps: float,
        on_stop_requested: "Any",
        plot_width: int = 420,
        gripper_calib: tuple[np.ndarray, np.ndarray] | None = None,
        gripper_max_width_mm: float = GRIPPER_MAX_WIDTH_MM,
        on_start_requested: "Any" = None,
        recording_active: "Any" = None,
        on_episode_stop_requested: "Any" = None,
    ):
        self.cv2 = cv2_module
        self.window_seconds = window_seconds
        self.min_interval_ns = int(1e9 / monitor_fps) if monitor_fps > 0 else 0
        self.plot_width = plot_width
        self.on_stop_requested = on_stop_requested
        self.gripper_calib = gripper_calib  # optional (positions, widths) -- see interp_gripper_width_mm
        self.gripper_max_width_mm = gripper_max_width_mm
        # Optional -- called (from this monitor thread) when Enter/Space is
        # pressed while the window has focus, same idea as on_stop_requested
        # for q/Esc/close. Without this, "Press Enter to start capture" only
        # listens on the terminal's stdin, so a keystroke sent to the GUI
        # window (which is what has OS keyboard focus while you're looking
        # at it) goes nowhere and just looks like recording never starts.
        # Calling it more than once (e.g. pressed again after recording
        # already started) is harmless -- see main()'s use of a
        # threading.Event, which is idempotent to repeat set() calls.
        self.on_start_requested = on_start_requested
        # Optional -- called (from this monitor thread) when Enter/Space is
        # pressed WHILE recording_active is set. Enter/Space is one button
        # that means different things depending on state: starts when idle
        # (on_start_requested above), ends the current episode when
        # recording (this one) -- mirrors the terminal's own "press Enter to
        # start" / "press Enter to stop" prompts. Without this, Enter/Space
        # in the monitor window only ever started recording and never
        # stopped it -- pressing it again mid-recording did nothing (only
        # q/Esc/close, via on_stop_requested, actually stopped anything, and
        # that also aborts the whole batch, not just this episode).
        self.on_episode_stop_requested = on_episode_stop_requested
        # Optional -- the same threading.Event RealSenseCapture/ServoPoller
        # already use to gate whether frames/samples actually get recorded
        # (capture.active). Read-only here, both to show the recording
        # status and to decide what Enter/Space should do (see above); this
        # class never sets or clears it.
        self.recording_active = recording_active
        self.times: list[float] = []
        self.positions: list[float] = []
        self.total_samples_seen = 0  # monotonically increasing, unlike len(self.times),
        # which drops old points as the rolling window slides -- this is
        # what proves data is still flowing even once the window is full.
        self.t0 = time.monotonic()
        self._panel_buffer: Any | None = None  # allocated once, reused every redraw
        self._pending_lock = threading.Lock()
        self._pending_frame: Any | None = None
        self._pending_servo: dict[str, Any] | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_pending(self, frame_rgb: Any | None, servo_sample: dict[str, Any] | None) -> None:
        """Cheap, non-blocking: called from the capture thread on every
        frame. Just remembers the latest values for the monitor thread to
        pick up on its own schedule."""
        with self._pending_lock:
            if frame_rgb is not None:
                self._pending_frame = frame_rgb
            if servo_sample is not None:
                self._pending_servo = servo_sample

    def _render_plot_panel(self, height: int) -> Any:
        cv2 = self.cv2
        if self._panel_buffer is None or self._panel_buffer.shape[0] != height:
            self._panel_buffer = np.empty((height, self.plot_width, 3), dtype=np.uint8)
        panel = self._panel_buffer
        panel[:] = self.BG_COLOR
        margin = 44
        plot_h = max(1, height - 2 * margin)
        plot_w = max(1, self.plot_width - 2 * margin)

        # Grid, drawn first so the title/line/text render on top of it.
        n = self.GRID_DIVISIONS
        for i in range(n + 1):
            y = margin + round(i * plot_h / n)
            cv2.line(panel, (margin, y), (margin + plot_w, y), self.GRID_COLOR, 1, cv2.LINE_AA)
        for i in range(n + 1):
            x = margin + round(i * plot_w / n)
            cv2.line(panel, (x, margin), (x, margin + plot_h), self.GRID_COLOR, 1, cv2.LINE_AA)

        cv2.putText(panel, "Servo position (deg)", (margin, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR, 1, cv2.LINE_AA)

        # Recording status -- text version of the border _redraw_once draws
        # around the whole window, so it's readable even if the border color
        # alone is hard to tell apart (e.g. on a dim/glare-y screen).
        is_recording = self.recording_active is not None and self.recording_active.is_set()
        status_text = "* RECORDING (Enter/Space to stop)" if is_recording else "o waiting to start (press Enter/Space)"
        status_color = self.RECORDING_COLOR if is_recording else self.IDLE_COLOR
        cv2.putText(panel, status_text, (margin, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2 if is_recording else 1, cv2.LINE_AA)

        if len(self.times) >= 2:
            t0, t1 = self.times[0], self.times[-1]
            p_min, p_max = min(self.positions), max(self.positions)
            if p_max - p_min < 1e-6:
                p_min, p_max = p_min - 1.0, p_max + 1.0
            if t1 - t0 < 1e-6:
                t1 = t0 + 1.0

            # Y-axis tick labels at each horizontal gridline, so the grid is
            # actually readable against real position values, not just
            # decorative lines.
            for i in range(n + 1):
                y = margin + round(i * plot_h / n)
                value = p_max - (i / n) * (p_max - p_min)
                cv2.putText(panel, f"{value:.1f}", (2, y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.TEXT_COLOR, 1, cv2.LINE_AA)

            # Vectorized instead of a per-point Python function call -- this
            # runs on every redraw and the rolling window can hold hundreds
            # of points at higher servo poll rates / longer --monitor-window.
            times_arr = np.asarray(self.times)
            positions_arr = np.asarray(self.positions)
            xs = margin + (times_arr - t0) / (t1 - t0) * plot_w
            ys = margin + plot_h - (positions_arr - p_min) / (p_max - p_min) * plot_h
            pts = np.column_stack([xs, ys]).astype(np.int32)
            cv2.polylines(panel, [pts], isClosed=False, color=self.LINE_COLOR, thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(panel, f"max {p_max:.1f}", (margin, margin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_COLOR, 1, cv2.LINE_AA)
            cv2.putText(panel, f"min {p_min:.1f}", (margin, height - margin + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_COLOR, 1, cv2.LINE_AA)

        last_str = f"{self.positions[-1]:.2f} deg" if self.positions else "no data yet"
        text_y = height - 14
        if self.gripper_calib is not None and self.positions:
            width_mm = interp_gripper_width_mm(self.positions[-1], self.gripper_calib, self.gripper_max_width_mm)
            cv2.putText(panel, f"gripper width: {width_mm:.1f} mm", (margin, text_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{self.total_samples_seen} pts, last={last_str}", (margin, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.TEXT_COLOR, 1, cv2.LINE_AA)
        return panel

    def _run(self) -> None:
        cv2 = self.cv2
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        try:
            while not self._stop_event.is_set():
                self._redraw_once()
                if self._stop_event.is_set():
                    break
                # Sleep until the next redraw is due instead of busy-looping;
                # also caps how often we even bother acquiring the lock.
                time.sleep(max(0.0, self.min_interval_ns / 1e9))
        finally:
            try:
                cv2.destroyWindow(self.WINDOW_NAME)
            except Exception:
                pass

    def _redraw_once(self) -> None:
        cv2 = self.cv2
        with self._pending_lock:
            frame_rgb = self._pending_frame
            servo_sample = self._pending_servo

        if servo_sample is not None:
            t = time.monotonic() - self.t0
            self.times.append(t)
            self.positions.append(servo_sample["position_deg"])
            self.total_samples_seen += 1
            cutoff = t - self.window_seconds
            while self.times and self.times[0] < cutoff:
                self.times.pop(0)
                self.positions.pop(0)

        if frame_rgb is None:
            return

        frame_bgr = frame_rgb[:, :, ::-1]  # RGB -> BGR view, no copy needed
        plot_panel = self._render_plot_panel(frame_bgr.shape[0])
        combined = np.hstack([frame_bgr, plot_panel])  # the one necessary copy

        # Border around the whole window -- red while actually recording,
        # gray otherwise -- so recording status is obvious at a glance
        # without having to read the status text (see _render_plot_panel).
        # Safe to draw on combined directly: hstack always makes a copy, so
        # this never touches the original pending frame buffer.
        is_recording = self.recording_active is not None and self.recording_active.is_set()
        border_color = self.RECORDING_COLOR if is_recording else self.IDLE_COLOR
        cv2.rectangle(combined, (0, 0), (combined.shape[1] - 1, combined.shape[0] - 1),
                      border_color, self.STATUS_BORDER_PX)

        cv2.imshow(self.WINDOW_NAME, combined)
        key = cv2.waitKey(1) & 0xFF
        window_closed = cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        if key in (27, ord("q")) or window_closed:
            self._stop_event.set()
            if self.on_stop_requested is not None:
                self.on_stop_requested()
        # 13/10 = Enter (backend-dependent which one waitKey reports), 32 =
        # Space -- lets you start/stop without having to click back onto the
        # terminal first. Context-sensitive, same as the terminal's own
        # prompts: starts while idle, ends the current episode while
        # recording -- see on_episode_stop_requested's docstring in
        # __init__ for why this needs to be a separate branch from just
        # always calling on_start_requested.
        elif key in (13, 10, 32):
            is_recording = self.recording_active is not None and self.recording_active.is_set()
            if is_recording:
                if self.on_episode_stop_requested is not None:
                    self.on_episode_stop_requested()
            elif self.on_start_requested is not None:
                self.on_start_requested()

    def close(self) -> None:
        """Ask the monitor thread to stop and wait briefly for it -- safe to
        call more than once."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def match_servo_to_frames(
    frame_rows: list[dict[str, Any]], servo_samples: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Match each recorded camera frame to the nearest-in-time servo sample.

    Uses host_time_ns for BOTH sides (not the RealSense device clock that
    camera_collecting.py's own match_samples_to_frames uses for IMU) -- the
    servo has no clock shared with the camera, so the host wall clock is the
    only timeline the two devices have in common. See module docstring.
    """
    if not servo_samples:
        return []
    sample_ts = [s["host_time_ns"] for s in servo_samples]
    matched: list[dict[str, Any]] = []
    for frame in frame_rows:
        frame_ts = frame["host_time_ns"]
        best_idx = cam.nearest_index_by_timestamp(sample_ts, frame_ts)
        sample = servo_samples[best_idx]
        matched.append(
            {
                "frame_id": frame["frame_id"],
                "frame_host_time_ns": frame_ts,
                "servo_host_time_ns": sample["host_time_ns"],
                "sync_error_ms": abs(sample["host_time_ns"] - frame_ts) / 1e6,
                "position_deg": sample["position_deg"],
                "velocity_deg_s": sample["velocity_deg_s"],
                "current_ma": sample["current_ma"],
            }
        )
    return matched


def load_gripper_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a servo-angle -> gripper-opening-width calibration table.

    Expects a CSV with a header row containing (at least) the columns
    `gear_displacement` (deg) and `gripper_displacement` (mm) -- one row per
    calibration point, any order (sorted here so np.interp gets a monotonic
    x). See DEFAULT_GRIPPER_CALIBRATION_CSV. Used to turn a recorded servo
    angle into a physical gripper opening width -- see
    interp_gripper_width_mm.
    """
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Gripper calibration CSV is empty: {path}")
    missing = {"gear_displacement", "gripper_displacement"} - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"Gripper calibration CSV {path} is missing column(s) {sorted(missing)}; "
            f"expected a header with gear_displacement, gripper_displacement."
        )
    if len(rows) < 2:
        raise ValueError(f"Gripper calibration CSV {path} needs at least 2 rows to interpolate, got {len(rows)}.")
    pairs = sorted((float(r["gear_displacement"]), float(r["gripper_displacement"])) for r in rows)
    positions = np.array([p for p, _ in pairs])
    widths = np.array([w for _, w in pairs])
    return positions, widths


def interp_gripper_width_mm(
    position_deg: float,
    calib: tuple[np.ndarray, np.ndarray],
    max_width_mm: float = GRIPPER_MAX_WIDTH_MM,
) -> float:
    """Linearly interpolate gripper opening width (mm) for a servo angle
    (deg). Positions outside the calibration table's own range are clamped
    to the nearest endpoint's width (np.interp's default behavior) rather
    than extrapolated -- a position the gripper was never actually
    calibrated at shouldn't produce a made-up width. The result is then
    additionally capped at max_width_mm -- see GRIPPER_MAX_WIDTH_MM -- since
    the CSV's model overshoots the real physical max near full-open."""
    positions, widths = calib
    width = float(np.interp(position_deg, positions, widths))
    return min(width, max_width_mm)


def apply_gripper_calibration(
    matched_rows: list[dict[str, Any]],
    calib: tuple[np.ndarray, np.ndarray] | None,
    max_width_mm: float = GRIPPER_MAX_WIDTH_MM,
) -> None:
    """Annotate each matched servo row in place with gripper_width_mm --
    None when no calibration table was given (--gripper-calibration-csv),
    so the CSV column always exists with a consistent schema either way."""
    for row in matched_rows:
        row["gripper_width_mm"] = (
            interp_gripper_width_mm(row["position_deg"], calib, max_width_mm) if calib is not None else None
        )


def write_camera_trajectory_csv(path: Path, trajectory_rows: list[dict[str, Any]]) -> None:
    """One row per tracked camera frame this episode -- used for both
    trackers' output (see _on_stereo_frame_for_orbslam and
    _on_frame_for_openvins in main()), just with different trajectory_rows
    lists/output paths (see main()'s write step). Core schema
    (timestamp,x,y,z,q_x,q_y,q_z,q_w,is_lost) matches what's used elsewhere
    for VIO/SLAM trajectory output in this project (see
    open_vins/ov_msckf/src/run_folder_vio.cpp), so downstream tooling can
    treat a live-recorded episode the same as an offline-processed one,
    regardless of which backend tracked it. is_lost=1 rows (before the
    tracker has initialized, or after it lost tracking) carry all-zero
    position/identity-ish quaternion placeholders, same convention as
    run_folder_vio.cpp.

    map_epoch (ORB-SLAM3 only, blank for OpenVINS -- no equivalent concept):
    System::GetMapEpoch() as of that row, from OrbSlamTracker.last_map_epoch
    (see its docstring). Two rows sharing the same map_epoch are in the same
    coordinate frame; a map_epoch CHANGE between consecutive rows means
    ORB-SLAM3 just reset in a way that can move the active map's pose origin
    (observed live to happen multiple times per recording, not just at
    startup -- e.g. "Not enough motion for initializing. Reseting..." or
    "IMU is not or recently initialized. Reseting active map...") and the
    poses on either side of that change must NOT be treated as one
    continuous trajectory -- downstream consumers should split (or discard)
    episodes at a map_epoch boundary rather than silently training across a
    coordinate-frame teleport. NOT the active map's raw ID: live testing
    showed the common reset path clears and re-initializes the SAME map
    object in place (its own ID never changes even though the origin does)
    -- see System.h's GetMapEpoch() doc comment for the full story."""
    fieldnames = ["timestamp", "x", "y", "z", "q_x", "q_y", "q_z", "q_w", "is_lost", "map_epoch"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trajectory_rows)


def write_servo_csv(path: Path, matched_rows: list[dict[str, Any]]) -> None:
    # Only frame/timing (to keep this aligned with servo.csv's usual role as
    # a per-stream file matching gyro.csv/accel.csv) plus gripper_width_mm --
    # raw position_deg/velocity_deg_s/current_ma/sync_error_ms are no longer
    # written here (recording only needs camera + gripper width now), even
    # though each row in matched_rows still carries them internally (used by
    # apply_gripper_calibration to compute gripper_width_mm in the first
    # place) -- extrasaction="ignore" lets DictWriter silently skip those
    # extra keys instead of erroring on them.
    fieldnames = [
        "frame_id",
        "frame_host_time_ns",
        "servo_host_time_ns",
        "gripper_width_mm",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(matched_rows)


def write_synced_csv(
    path: Path,
    frame_rows: list[dict[str, Any]],
    gyro_matched: list[dict[str, Any]],
    accel_matched: list[dict[str, Any]],
    servo_matched: list[dict[str, Any]],
) -> None:
    """One row per camera frame, joining every stream by frame_id -- the
    single table most downstream uses (e.g. an imitation-learning dataset)
    actually want, on top of the individual per-stream CSVs this script also
    writes (kept for parity with camera_collecting.py's existing files)."""
    gyro_by_frame = {r["frame_id"]: r for r in gyro_matched}
    accel_by_frame = {r["frame_id"]: r for r in accel_matched}
    servo_by_frame = {r["frame_id"]: r for r in servo_matched}

    fieldnames = [
        "frame_id",
        "host_time_ns",
        "color_timestamp_seconds",
        "rgb_path",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "gyro_sync_error_ms",
        "accel_x",
        "accel_y",
        "accel_z",
        "accel_sync_error_ms",
        "servo_gripper_width_mm",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for frame in frame_rows:
            fid = frame["frame_id"]
            gyro = gyro_by_frame.get(fid)
            accel = accel_by_frame.get(fid)
            servo = servo_by_frame.get(fid)
            writer.writerow(
                {
                    "frame_id": fid,
                    "host_time_ns": frame["host_time_ns"],
                    "color_timestamp_seconds": frame["color_timestamp_seconds"],
                    "rgb_path": frame["rgb_path"],
                    "gyro_x": gyro["x"] if gyro else None,
                    "gyro_y": gyro["y"] if gyro else None,
                    "gyro_z": gyro["z"] if gyro else None,
                    "gyro_sync_error_ms": (
                        abs(gyro["timestamp_seconds"] - frame["color_timestamp_seconds"]) * 1000.0
                        if gyro
                        else None
                    ),
                    "accel_x": accel["x"] if accel else None,
                    "accel_y": accel["y"] if accel else None,
                    "accel_z": accel["z"] if accel else None,
                    "accel_sync_error_ms": (
                        abs(accel["timestamp_seconds"] - frame["color_timestamp_seconds"]) * 1000.0
                        if accel
                        else None
                    ),
                    "servo_gripper_width_mm": servo["gripper_width_mm"] if servo else None,
                }
            )


def summarize_sync(label: str, matched_rows: list[dict[str, Any]]) -> None:
    if not matched_rows:
        print(f"  {label}: no samples recorded")
        return
    errors = [r["sync_error_ms"] for r in matched_rows]
    print(
        f"  {label}: mean sync error {sum(errors) / len(errors):.1f} ms, "
        f"max {max(errors):.1f} ms, over {len(matched_rows)} frames"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Camera
    parser.add_argument("--data-dir", type=Path, default=cam.DEFAULT_DATA_DIR)
    parser.add_argument("--max-duration", type=float, default=None,
                         help="Optional safety cap in seconds; capture auto-stops if reached.")
    parser.add_argument("--queue-size", type=int, default=256)
    # Defaulting width/height too (not just fps) so this fully specifies a
    # profile by default -- cam.optional_profile() only honors --color-fps
    # when width/height are ALSO given; otherwise it falls back to
    # camera_collecting.py's own auto-negotiation list, which tries 60fps
    # first. 30fps is the default here (not 60) because at 60fps the
    # per-frame PNG encode + disk write in _save_frameset has too little
    # time budget (16.7ms) and the recording pipeline falls behind
    # real time -- unrelated to the live monitor, which is why reworking the
    # monitor never fixed the delay. Pass --color-fps 60 explicitly (with
    # --color-width/--color-height) if the write path can keep up on your
    # hardware.
    parser.add_argument("--color-width", type=int, default=848)
    parser.add_argument("--color-height", type=int, default=480)
    parser.add_argument("--color-fps", type=int, default=30)
    parser.add_argument("--depth-flag", dest="depth_flag", action=argparse.BooleanOptionalAction, default=False,
                         help="Save depth images too (default: off -- this script's default output is "
                              "RGB + IMU + servo position, per the intended use of this script).")
    parser.add_argument("--imu-fps", type=int, default=200, dest="imu_fps")

    # Servo (mirrors servo/spring_position_mode.py's flags)
    parser.add_argument("--servo-port", default=None,
                         help="Serial port for the U2D2. Auto-detected if omitted, same as scan_servo.py.")
    parser.add_argument("--servo-baud", type=int, default=57600, dest="servo_baud")
    parser.add_argument("--servo-id", type=int, default=1, dest="servo_id")
    parser.add_argument("--p-gain", type=int, default=100, dest="p_gain")
    parser.add_argument("--i-gain", type=int, default=30, dest="i_gain")
    parser.add_argument("--d-gain", type=int, default=1200, dest="d_gain")
    parser.add_argument("--max-current", type=float, default=150.0, dest="max_current")
    parser.add_argument("--center", type=float, default=None,
                         help="Spring center offset in degrees from the starting position. Default: "
                              "pinned to the calibrated fully-open angle (see --gripper-open-deg), same "
                              "default as spring_position_mode.py.")
    parser.add_argument("--gripper-open-deg", type=float, default=servo_mod.GRIPPER_OPEN_DEG, dest="gripper_open_deg")
    parser.add_argument("--gripper-closed-deg", type=float, default=servo_mod.GRIPPER_CLOSED_DEG, dest="gripper_closed_deg")
    parser.add_argument("--range-margin", type=float, default=servo_mod.GRIPPER_RANGE_MARGIN_DEG, dest="range_margin")
    parser.add_argument("--ignore-range-limit", action="store_true", dest="ignore_range_limit")

    # Gripper calibration -- CSV mapping recorded servo angle to a physical
    # gripper opening width (mm), on by default now that
    # mapping_csv/mapping_function.csv exists. See load_gripper_calibration.
    parser.add_argument("--gripper-calibration-csv", type=Path, default=DEFAULT_GRIPPER_CALIBRATION_CSV,
                         dest="gripper_calibration_csv",
                         help="CSV with columns gear_displacement (deg), gripper_displacement (mm) -- one row "
                              f"per calibration point, any order. Default: {DEFAULT_GRIPPER_CALIBRATION_CSV} "
                              "-- adds a gripper_width_mm column to servo.csv/synced.csv (linearly interpolated, "
                              "clamped to the table's range) and shows the live width on the monitor window. "
                              "Pass --no-gripper-calibration to skip this entirely.")
    parser.add_argument("--no-gripper-calibration", dest="gripper_calibration_csv", action="store_const",
                         const=None,
                         help="Disable the gripper-width mapping (no gripper_width_mm column, no live readout).")
    parser.add_argument("--gripper-calibration-offset-deg", type=float, default=GRIPPER_CALIBRATION_BIAS_DEG,
                         dest="gripper_calibration_offset_deg",
                         help=f"Deg added to every gear_displacement value in --gripper-calibration-csv so it "
                              f"lines up with the servo's raw absolute reading. Default {GRIPPER_CALIBRATION_BIAS_DEG:.2f} "
                              f"(re-measured and ruler-verified 2026-08-26 after the gripper was physically "
                              f"reassembled -- see the module-level comment on GRIPPER_CALIBRATION_BIAS_DEG) -- "
                              f"re-measure and override this if either changes.")
    parser.add_argument("--gripper-max-width-mm", type=float, default=GRIPPER_MAX_WIDTH_MM,
                         dest="gripper_max_width_mm",
                         help=f"Real measured max gripper opening (mm) -- interpolated widths are capped at this "
                              f"value, since the calibration CSV's kinematic model overshoots it near full-open. "
                              f"Default {GRIPPER_MAX_WIDTH_MM:.1f} (measured 2026-08-19).")

    # Live ORB-SLAM3 (Stereo-Inertial) tracking -- see orbslam_bridge.py. This
    # is the primary/default live tracker (see module-level comment above
    # DEFAULT_OPENVINS_CONFIG for why). Needs stereo IR, so enabling this
    # also turns on --ir-flag-equivalent stereo capture on RealSenseCapture
    # (see record_ir=args.orbslam below) -- there's no separate --ir-flag
    # here, since IR is only ever needed for ORB-SLAM3 in this script. Fed
    # the full native-rate IMU stream (via on_imu_sample, paired
    # nearest-gyro-to-accel same as OpenVINS's own fusion) and every stereo
    # frame pair (via RealSenseCapture's on_stereo_frame hook). A fresh
    # tracker is created per episode (System has no in-place reset exposed
    # here -- see orbslam_bridge.py's docstring), so each episode's
    # camera_trajectory.csv starts its own clean map/pose graph near t=0.
    parser.add_argument("--orbslam", dest="orbslam", action=argparse.BooleanOptionalAction, default=True,
                         help="Run live ORB-SLAM3 Stereo-Inertial tracking alongside recording (default: on, "
                              "the primary live tracker -- see --openvins for the deprecated/comparison-only "
                              "monocular alternative). Prints tracked position periodically and writes "
                              "camera_trajectory.csv per episode. Also enables stereo IR capture (ir_left/ "
                              "ir_right). Use --no-orbslam to skip entirely (e.g. if ORB_SLAM3 isn't built).")
    parser.add_argument("--orbslam-settings", type=Path, default=ORBSLAM_DEFAULT_SETTINGS_PATH, dest="orbslam_settings",
                         help=f"ORB-SLAM3 settings YAML (calibration) to use. Default: {ORBSLAM_DEFAULT_SETTINGS_PATH}")
    parser.add_argument("--orbslam-vocab", type=Path, default=ORBSLAM_DEFAULT_VOCAB_PATH, dest="orbslam_vocab",
                         help=f"ORB-SLAM3 vocabulary file. Default: {ORBSLAM_DEFAULT_VOCAB_PATH}")
    parser.add_argument("--orbslam-lib", type=Path, default=ORBSLAM_DEFAULT_LIB_PATH, dest="orbslam_lib",
                         help=f"Path to the built liborb_capi.so. Default: {ORBSLAM_DEFAULT_LIB_PATH}")
    parser.add_argument("--orbslam-viewer", dest="orbslam_viewer", action=argparse.BooleanOptionalAction,
                         default=False,
                         help="Also launch ORB-SLAM3's own Pangolin 3D map viewer window (default: off -- "
                              "this script's own --monitor window already gives live feedback; the Pangolin "
                              "viewer is mainly for standalone debugging of tracking/map quality).")

    # Live OpenVINS (monocular) tracking -- see openvins_bridge.py. Off by
    # default now (see module-level comment above DEFAULT_OPENVINS_CONFIG);
    # kept only for side-by-side comparison against --orbslam. When both are
    # enabled, ORB-SLAM3's output is the canonical camera_trajectory.csv and
    # OpenVINS's goes to camera_trajectory_openvins.csv instead (see main()).
    parser.add_argument("--openvins", dest="openvins", action=argparse.BooleanOptionalAction, default=False,
                         help="Also run live (monocular) OpenVINS tracking, for comparison against --orbslam "
                              "(default: off). Use --no-orbslam --openvins to run OpenVINS alone, old-style.")
    parser.add_argument("--openvins-config", type=Path, default=DEFAULT_OPENVINS_CONFIG, dest="openvins_config",
                         help=f"OpenVINS estimator_config.yaml to use. Default: {DEFAULT_OPENVINS_CONFIG}")
    parser.add_argument("--openvins-lib", type=Path, default=OPENVINS_DEFAULT_LIB_PATH, dest="openvins_lib",
                         help=f"Path to the built libov_capi.so. Default: {OPENVINS_DEFAULT_LIB_PATH}")

    # Monitor -- one cv2 window: camera frame + servo position plot
    # composited side by side. See CombinedMonitor's docstring.
    parser.add_argument("--monitor", dest="monitor", action=argparse.BooleanOptionalAction, default=True,
                         help="Show the live camera+servo-position window while recording (default: on). "
                              "Press 'q' or close the window to stop recording early.")
    parser.add_argument("--monitor-fps", type=float, default=15.0, dest="monitor_fps",
                         help="Monitor window redraw rate, decoupled from the recording rate (default 15Hz).")
    parser.add_argument("--monitor-window", type=float, default=10.0, dest="monitor_window",
                         help="Seconds of servo-position history shown in the live plot at once.")

    # Batch recording -- e.g. collecting N demonstrations for imitation
    # learning back to back. See the episode loop in main().
    parser.add_argument("--episodes", type=int, default=1, dest="episodes",
                         help="Number of episodes to record in one session (default: 1, i.e. today's "
                              "single-recording behavior). The camera connection and servo torque stay up "
                              "the whole session -- only each episode's own state resets between episodes -- "
                              "so episode 2 starts instantly instead of paying reconnect/re-home cost again. "
                              "Each episode gets its own output folder and still needs its own Enter/Space "
                              "to start and stop, since a human is doing a variable-length demonstration each "
                              "time. Press q/Esc or close the monitor window to stop the whole batch early "
                              "(a plain Enter/Space only ends the current episode and moves on to the next).")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rs = cam.import_realsense()

    gripper_calib: tuple[np.ndarray, np.ndarray] | None = None
    if args.gripper_calibration_csv is not None:
        # Loaded, offset, and validated up front, before touching any
        # hardware -- a malformed calibration CSV should fail fast, not
        # after the servo's already been torqued on and a scan directory
        # created. The CSV's own servo_position_deg column is in whatever
        # frame it was measured in (e.g. starting from 0), while the servo
        # reports raw absolute encoder degrees (read as signed -- see
        # read_position_deg), so --gripper-calibration-offset-deg (a fixed,
        # measured constant -- see GRIPPER_CALIBRATION_BIAS_DEG) shifts the
        # table into that same raw frame.
        raw_positions, widths = load_gripper_calibration(args.gripper_calibration_csv)
        gripper_calib = (raw_positions + args.gripper_calibration_offset_deg, widths)
        print(f"Loaded gripper calibration: {len(gripper_calib[0])} points from "
              f"{args.gripper_calibration_csv}, offset {args.gripper_calibration_offset_deg:.1f} deg -> "
              f"effective range {gripper_calib[0].min():.1f}-{gripper_calib[0].max():.1f} deg -> "
              f"{gripper_calib[1].min():.1f}-{gripper_calib[1].max():.1f} mm")

    port = servo_mod.resolve_port(args.servo_port)
    servo = servo_mod.DynamixelPositionSpring(port, args.servo_baud, args.servo_id)
    # Declared before the try block (not just inside it) so `finally` can
    # always safely check them, even if something fails before either one
    # gets constructed.
    poller: ServoPoller | None = None
    monitor: CombinedMonitor | None = None
    orb_tracker: OrbSlamTracker | None = None
    orb_worker: OrbSlamWorker | None = None
    ov_tracker: OpenVinsTracker | None = None

    try:
        start_angle = servo.read_position_deg()
        if args.center is not None:
            center = start_angle + args.center
        else:
            center = args.gripper_open_deg
        if not args.ignore_range_limit:
            safe_min = min(args.gripper_open_deg, args.gripper_closed_deg) + args.range_margin
            safe_max = max(args.gripper_open_deg, args.gripper_closed_deg) - args.range_margin
            if center < safe_min or center > safe_max:
                clamped = min(max(center, safe_min), safe_max)
                print(f"Warning: spring center {center:.2f} deg outside safe range "
                      f"[{safe_min:.2f}, {safe_max:.2f}] deg -- clamping to {clamped:.2f} deg.")
                center = clamped

        print(f"Servo: starting angle {start_angle:.2f} deg, spring center {center:.2f} deg, "
              f"P={args.p_gain} I={args.i_gain} D={args.d_gain}, max current {args.max_current} mA")
        servo.configure_spring(args.p_gain, args.i_gain, args.d_gain, args.max_current, center)
        print("Servo torque enabled.")

        args.data_dir.mkdir(parents=True, exist_ok=True)

        requested_color = cam.optional_profile(args.color_width, args.color_height, args.color_fps, "color")

        # First episode's folder has to exist before RealSenseCapture is
        # constructed (scan_dir is baked in at construction time). Later
        # episodes swap it out via capture.reset_for_new_episode instead of
        # rebuilding the camera pipeline from scratch -- see that method's
        # docstring for why a fresh pipeline.start() per episode is worth
        # avoiding (it's the slow part; everything else resets cheaply).
        scan_dir = cam.next_scan_dir(args.data_dir)
        scan_dir.mkdir()

        capture = cam.RealSenseCapture(
            rs=rs,
            scan_dir=scan_dir,
            max_duration_seconds=args.max_duration,
            requested_color=requested_color,
            requested_depth=None,
            enable_imu=True,
            record_rgb=True,
            record_depth=args.depth_flag,
            # ORB-SLAM3 Stereo-Inertial needs left/right IR -- there's no
            # separate --ir-flag on this script since IR is only ever needed
            # here for --orbslam. requested_ir left as None (auto-negotiate
            # from camera_collecting.py's IR_PROFILES) since --orbslam-settings'
            # calibration was measured at that same auto-negotiated 848x480
            # default (see ORB_SLAM3/config/RealSense_D435i_ours.yaml).
            record_ir=args.orbslam,
            queue_size=args.queue_size,
            imu_fps=args.imu_fps,
            # Not using RealSenseCapture's own cv2 preview window -- Combined
            # Monitor builds its own single window (camera + servo plot)
            # instead, fed via on_preview_frame below. preview_fps still
            # throttles how often on_preview_frame fires (see _update_preview).
            show_preview=False,
            preview_fps=args.monitor_fps,
        )

        # ServoPoller also stays up the whole session (own USB-paced thread,
        # nothing scan_dir-specific about it) -- only poller.samples gets
        # reset per episode, right before that episode starts recording.
        poller = ServoPoller(servo, capture.active)
        poller.start()

        # Set by whichever comes first: Enter on the terminal, or Enter/Space
        # in the monitor window (see CombinedMonitor's on_start_requested).
        # Needed because the GUI window -- not the terminal -- has OS
        # keyboard focus while you're actually looking at it, so a keystroke
        # sent there never reached the terminal's stdin before this: pressing
        # Enter "in the monitor window" looked exactly like the recording
        # just never starting. Reassigned to a fresh Event each episode
        # (below); on_stop_requested references it by name (same enclosing
        # scope), not by value, so it always signals whichever episode's
        # start_event is current -- see _on_monitor_closed's comment.
        start_event = threading.Event()

        # Distinct from a per-episode stop: q/Esc/closing the monitor window
        # means "stop the whole batch", not just "end this one episode and
        # prompt for the next" -- a plain Enter/Space (terminal or monitor)
        # only ends the current episode.
        abort_batch_event = threading.Event()

        # Needed for both the monitor window AND OpenVINS's RGB->gray
        # conversion below -- imported once, up front, regardless of which
        # one(s) are actually enabled, so --no-monitor --openvins still works.
        cv2_module = cam.import_opencv() if (args.monitor or args.openvins or args.orbslam) else None

        if args.monitor:
            try:
                cv2 = cv2_module

                def _on_monitor_closed() -> None:
                    print("\nMonitor window closed/q pressed -- stopping this episode and the batch.")
                    capture.stop_event.set()
                    abort_batch_event.set()
                    # Also unblocks a start_event.wait() if this fires while
                    # still waiting for an episode to start (not yet
                    # recording) -- without this, q/close before ever
                    # pressing Enter would leave the script hanging forever
                    # instead of actually stopping.
                    start_event.set()

                monitor = CombinedMonitor(
                    cv2,
                    window_seconds=args.monitor_window,
                    monitor_fps=args.monitor_fps,
                    on_stop_requested=_on_monitor_closed,
                    gripper_calib=gripper_calib,
                    gripper_max_width_mm=args.gripper_max_width_mm,
                    on_start_requested=start_event.set,
                    recording_active=capture.active,
                    # Constant across episodes (unlike on_start_requested,
                    # which gets rebound to a fresh per-episode Event below)
                    # -- capture.stop_event itself gets swapped out each
                    # episode by reset_for_new_episode, but this looks it up
                    # fresh at call time via attribute access, so it always
                    # targets whichever episode is currently running. Ends
                    # just this episode, NOT the whole batch (that's
                    # q/Esc/close, via on_stop_requested above).
                    on_episode_stop_requested=lambda: capture.stop_event.set(),
                )
                monitor.start()
            except Exception as exc:
                print(f"--monitor requested but the window could not be created ({exc}); continuing without it.")
                monitor = None

        def _on_preview_frame(color: Any) -> None:
            # Fires regardless of recording state (see camera_collecting.py's
            # _frame_callback) -- this is what makes the monitor show real
            # content from the moment the camera starts, not only once you
            # press Enter to begin recording. Must stay cheap AND non-
            # blocking: this runs on librealsense's own internal SDK
            # callback thread, not ours, so even a brief lock wait here
            # risks stalling frame delivery. A non-blocking acquire means
            # the worst case is this one frame's servo reading is skipped
            # (poller.latest is updated many times a second, so the next
            # frame picks it up) rather than ever blocking the SDK thread.
            if monitor is not None:
                latest = None
                if poller.lock.acquire(blocking=False):
                    try:
                        latest = poller.latest
                    finally:
                        poller.lock.release()
                monitor.set_pending(color, latest)

        capture.on_preview_frame = _on_preview_frame

        capture.start_camera()
        capture.print_camera_summary()
        print(f"Servo poll thread running (target: as fast as the USB link allows).")
        if args.episodes > 1:
            print(f"\nBatch mode: recording up to {args.episodes} episodes. Press q/Esc or close the "
                  f"monitor window at any point to stop the whole batch early.")

        episode_results: list[dict[str, Any]] = []

        # OpenVINS has no in-place reset (VioManager tracks continuously once
        # built) -- a fresh tracker is created per episode below so each
        # episode's camera_trajectory.csv starts its own clean init near t=0,
        # matching how the offline run_folder_vio processes each scan folder
        # independently. None across the whole batch if --no-openvins.
        # (declared above the try block, alongside poller/monitor, so the
        # finally clause can always safely close it)

        for episode_idx in range(1, args.episodes + 1):
            if episode_idx > 1:
                scan_dir = cam.next_scan_dir(args.data_dir)
                scan_dir.mkdir()
                capture.reset_for_new_episode(scan_dir)
                with poller.lock:
                    poller.samples = []

            # Worker closed (thread joined) BEFORE the tracker it wraps is
            # closed/destroyed -- see OrbSlamWorker.close's docstring for why
            # the ordering matters (no in-flight track_stereo against a freed
            # handle).
            if orb_worker is not None:
                orb_worker.close()
            if orb_tracker is not None:
                orb_tracker.close()
            orb_tracker = (
                OrbSlamTracker(args.orbslam_settings, args.orbslam_vocab, args.orbslam_lib,
                                use_viewer=args.orbslam_viewer)
                if args.orbslam
                else None
            )
            orb_trajectory: list[dict[str, Any]] = []
            orb_last_print = [0.0]  # mutable box, see _on_orb_pose
            orb_last_map_epoch: list[int | None] = [None]  # mutable box, see _on_orb_pose
            orb_map_resets = [0]  # mutable box, see _on_orb_pose

            def _on_orb_pose(
                ts: float,
                pose: tuple | None,
                map_epoch: int,
                trajectory: list[dict[str, Any]] = orb_trajectory,
                last_print: list[float] = orb_last_print,
                last_map_epoch: list[int | None] = orb_last_map_epoch,
                map_resets: list[int] = orb_map_resets,
            ) -> None:
                # Runs on OrbSlamWorker's own thread, not the capture thread
                # -- see that class's docstring. trajectory/last_print/
                # last_map_epoch/map_resets are bound as defaults at
                # definition time (same closure-capture pattern used
                # throughout this per-episode block) so this always targets
                # THIS episode's own state, even though a new
                # OrbSlamWorker/tracker gets created fresh each episode.
                #
                # map_epoch changing between calls means ORB-SLAM3 just
                # reset in a way that can move the active map's pose origin
                # -- e.g. after "Not enough motion for initializing.
                # Reseting..." or "IMU is not or recently initialized.
                # Reseting active map..." -- observed live to happen
                # MULTIPLE TIMES during a single ~20s recording, not just at
                # startup. Poses before and after such a change are NOT in
                # the same coordinate frame and must not be treated as a
                # continuous trajectory -- recorded here (map_epoch column)
                # rather than silently concatenated, so downstream
                # processing can split/discard segments instead of training
                # on a trajectory with invisible teleports in it. NOT simply
                # "the active map's ID changed" -- see write_camera_trajectory_csv's
                # map_epoch docstring for why a dedicated counter is needed
                # (the common reset path clears and reuses the same map
                # object, so its own ID never changes).
                if last_map_epoch[0] is not None and map_epoch != last_map_epoch[0]:
                    map_resets[0] += 1
                    print(f"\n[orbslam] WARNING: map reset detected (map_epoch {last_map_epoch[0]} -> {map_epoch}) "
                          f"at t={ts:.2f} -- trajectory is NOT continuous across this point")
                last_map_epoch[0] = map_epoch
                if pose is not None:
                    x, y, z, qx, qy, qz, qw = pose
                    trajectory.append(
                        {"timestamp": ts, "x": x, "y": y, "z": z, "q_x": qx, "q_y": qy, "q_z": qz, "q_w": qw,
                         "is_lost": 0, "map_epoch": map_epoch}
                    )
                    now = time.monotonic()
                    if now - last_print[0] >= 1.0:
                        last_print[0] = now
                        print(f"\n[orbslam] t={ts:.2f}  pos = {x:.3f}, {y:.3f}, {z:.3f}")
                else:
                    trajectory.append(
                        {"timestamp": ts, "x": 0.0, "y": 0.0, "z": 0.0, "q_x": 0.0, "q_y": 0.0, "q_z": 0.0, "q_w": 1.0,
                         "is_lost": 1, "map_epoch": map_epoch}
                    )

            orb_worker = OrbSlamWorker(orb_tracker, _on_orb_pose) if orb_tracker is not None else None
            # ORB-SLAM3's feed_imu wants one already-paired gyro+accel sample
            # per call (unlike OpenVinsTracker, which does its own gyro/accel
            # fusion internally) -- track the latest gyro reading here and
            # pair it with each accel sample, same nearest-neighbor fusion
            # run_realsense_vio.cpp/the stock D435i example both use for the
            # D435I's two independent gyro/accel streams.
            orb_last_gyro = [0.0, 0.0, 0.0]
            orb_have_gyro = [False]

            if ov_tracker is not None:
                ov_tracker.close()
            ov_tracker = OpenVinsTracker(args.openvins_config, args.openvins_lib) if args.openvins else None
            ov_trajectory: list[dict[str, Any]] = []
            ov_last_print = [0.0]  # mutable box, see _on_frame_for_openvins

            def _on_imu_sample(
                kind: str,
                row: dict[str, Any],
                ov: OpenVinsTracker | None = ov_tracker,
                orb: OrbSlamTracker | None = orb_tracker,
                last_gyro: list[float] = orb_last_gyro,
                have_gyro: list[bool] = orb_have_gyro,
            ) -> None:
                if kind == "gyro":
                    if ov is not None:
                        ov.feed_imu_gyro(row["x"], row["y"], row["z"])
                    last_gyro[0], last_gyro[1], last_gyro[2] = row["x"], row["y"], row["z"]
                    have_gyro[0] = True
                elif kind == "accel":
                    if ov is not None:
                        ov.feed_imu_accel(row["timestamp_seconds"], row["x"], row["y"], row["z"])
                    if orb is not None and have_gyro[0]:
                        # Host clock (time.time_ns(), captured the instant
                        # this sample arrived), NOT row["timestamp_seconds"]
                        # (RealSense's own per-sensor hardware timestamp) --
                        # see _on_stereo_frame_for_orbslam's matching comment
                        # for why: live testing found gyro/accel's hardware
                        # timestamp advancing roughly 2x faster than color's
                        # once IR streams were added alongside color+IMU,
                        # i.e. they are not reliably on the same clock domain
                        # in this stream combination. The host wall clock is
                        # the one timeline every stream shares by
                        # construction, same reasoning match_servo_to_frames
                        # already uses for the servo (which has no hardware
                        # clock in common with the camera at all).
                        orb.feed_imu(
                            row["host_time_ns"] / 1e9,
                            last_gyro[0], last_gyro[1], last_gyro[2],
                            row["x"], row["y"], row["z"],
                        )

            def _on_frame_for_openvins(
                frame_id: int,
                host_time_ns: int,
                color: Any,
                tracker: OpenVinsTracker | None = ov_tracker,
                trajectory: list[dict[str, Any]] = ov_trajectory,
                last_print: list[float] = ov_last_print,
            ) -> None:
                if tracker is None or color is None:
                    return
                ts = capture.frame_rows[-1]["color_timestamp_seconds"]
                gray = cv2_module.cvtColor(color, cv2_module.COLOR_RGB2GRAY)
                tracker.feed_camera_gray(ts, gray)
                pose = tracker.get_pose()
                if pose is not None:
                    x, y, z, qx, qy, qz, qw = pose
                    trajectory.append(
                        {"timestamp": ts, "x": x, "y": y, "z": z, "q_x": qx, "q_y": qy, "q_z": qz, "q_w": qw,
                         "is_lost": 0, "map_epoch": ""}  # no map-reset concept for OpenVINS -- see write_camera_trajectory_csv
                    )
                    now = time.monotonic()
                    if now - last_print[0] >= 1.0:
                        last_print[0] = now
                        print(f"\n[openvins] t={ts:.2f}  pos = {x:.3f}, {y:.3f}, {z:.3f}")
                else:
                    trajectory.append(
                        {"timestamp": ts, "x": 0.0, "y": 0.0, "z": 0.0, "q_x": 0.0, "q_y": 0.0, "q_z": 0.0, "q_w": 1.0,
                         "is_lost": 1, "map_epoch": ""}
                    )

            def _on_stereo_frame_for_orbslam(
                frame_id: int,
                host_time_ns: int,
                left_gray: Any,
                right_gray: Any,
                worker: OrbSlamWorker | None = orb_worker,
            ) -> None:
                # Cheap hand-off only -- see OrbSlamWorker's docstring for
                # why actual tracking happens on its own thread instead of
                # here (this runs on RealSenseCapture's capture-draining
                # thread, which must stay fast: it also gates how quickly
                # frames get saved to disk).
                if worker is None:
                    return
                # Host clock, not color_timestamp_seconds -- see the matching
                # comment in _on_imu_sample's accel branch. host_time_ns is
                # captured once per frame in _save_frameset (camera_collecting.py),
                # same value already used to sync the servo to frames, so
                # this and the IMU feed above are now both on that one shared
                # timeline.
                ts = capture.frame_rows[-1]["host_time_ns"] / 1e9
                worker.submit(ts, left_gray, right_gray)

            capture.on_imu_sample = _on_imu_sample
            capture.on_frame = _on_frame_for_openvins
            capture.on_stereo_frame = _on_stereo_frame_for_orbslam

            # Written into every episode's own folder (cheap -- unlike the
            # camera pipeline, this doesn't need to be shared/reused) so
            # each output folder is self-contained on its own, same as
            # today's single-episode behavior.
            cam.write_calibration(rs, capture.profile, scan_dir / "calibration.json", args.depth_flag)

            start_event = threading.Event()
            if monitor is not None:
                monitor.on_start_requested = start_event.set

            print(f"\n=== Episode {episode_idx}/{args.episodes} -- output: {scan_dir} ===")
            print("Press Enter to start capture (in this terminal, or in the monitor window if it has focus)...")

            def _wait_for_terminal_enter(event: threading.Event = start_event) -> None:
                try:
                    input()
                except EOFError:
                    pass  # stdin closed (e.g. non-interactive run) -- fall back to the monitor keypress only
                event.set()

            threading.Thread(target=_wait_for_terminal_enter, daemon=True).start()
            start_event.wait()

            if abort_batch_event.is_set():
                print("Batch stopped before this episode started recording.")
                break

            def _wait_for_stop(event: threading.Event = capture.stop_event) -> None:
                input("Recording... press Enter to stop.\n")
                event.set()

            threading.Thread(target=_wait_for_stop, daemon=True).start()

            capture_result = capture.capture()

            # Drain OrbSlamWorker before reading orb_trajectory below: it
            # runs on its own thread (see that class's docstring), so the
            # very last submitted frame may still be mid-track_stereo() at
            # the moment capture() returns. close() joins the thread (which
            # finishes whatever's in flight before exiting its loop), so
            # this guarantees orb_trajectory reflects every frame this
            # worker ever got to. Safe to call here even though the top of
            # the next loop iteration also calls it -- joining an
            # already-stopped thread is a no-op.
            if orb_worker is not None:
                orb_worker.close()

            poller_error = poller.error  # snapshot -- poller keeps running/retrying into the next episode
            if poller_error is not None:
                print(f"\nWarning: servo poller hit an error during this episode: {poller_error}")

            gyro_matched = cam.match_samples_to_frames(capture.frame_rows, capture.gyro_rows)
            accel_matched = cam.match_samples_to_frames(capture.frame_rows, capture.accel_rows)
            servo_matched = match_servo_to_frames(capture.frame_rows, poller.samples)
            apply_gripper_calibration(servo_matched, gripper_calib, args.gripper_max_width_mm)

            write_servo_csv(scan_dir / "servo.csv", servo_matched)
            write_synced_csv(scan_dir / "synced.csv", capture.frame_rows, gyro_matched, accel_matched, servo_matched)
            # ORB-SLAM3 is the canonical camera_trajectory.csv whenever it ran
            # (see module-level comment above DEFAULT_OPENVINS_CONFIG); when
            # OpenVINS also ran (--openvins, comparison mode) it gets its own
            # separate file instead of being silently dropped. If only
            # OpenVINS ran (--no-orbslam --openvins), it takes over the
            # canonical filename, preserving this script's old single-tracker
            # behavior.
            if orb_tracker is not None:
                write_camera_trajectory_csv(scan_dir / "camera_trajectory.csv", orb_trajectory)
                n_tracked = sum(1 for r in orb_trajectory if r["is_lost"] == 0)
                print(f"ORB-SLAM3: {n_tracked}/{len(orb_trajectory)} frames tracked, "
                      f"{orb_map_resets[0]} map reset(s) ({orb_map_resets[0] + 1} coordinate-frame segment(s)) "
                      f"-> wrote {scan_dir / 'camera_trajectory.csv'}")
                if orb_worker is not None:
                    print(orb_worker.stats_summary())
                if ov_tracker is not None:
                    write_camera_trajectory_csv(scan_dir / "camera_trajectory_openvins.csv", ov_trajectory)
                    n_tracked_ov = sum(1 for r in ov_trajectory if r["is_lost"] == 0)
                    print(f"OpenVINS (comparison): {n_tracked_ov}/{len(ov_trajectory)} frames tracked "
                          f"-> wrote {scan_dir / 'camera_trajectory_openvins.csv'}")
            elif ov_tracker is not None:
                write_camera_trajectory_csv(scan_dir / "camera_trajectory.csv", ov_trajectory)
                n_tracked = sum(1 for r in ov_trajectory if r["is_lost"] == 0)
                print(f"OpenVINS: {n_tracked}/{len(ov_trajectory)} frames tracked "
                      f"-> wrote {scan_dir / 'camera_trajectory.csv'}")

            print(f"\nServo poll rate achieved: {poller.achieved_hz:.1f} Hz "
                  f"({len(poller.samples)} samples recorded during the active window)")
            print("Sync quality (camera frame vs. matched sample):")
            summarize_sync("servo", servo_matched)

            metadata_path = scan_dir / "metadata.json"
            metadata: dict[str, Any] = {}
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
            metadata.update(
                {
                    "script": Path(__file__).name,
                    "episode_index": episode_idx,
                    "episodes_requested": args.episodes,
                    "servo": {
                        "port": port,
                        "baud": args.servo_baud,
                        "id": args.servo_id,
                        "p_gain": args.p_gain,
                        "i_gain": args.i_gain,
                        "d_gain": args.d_gain,
                        "max_current_ma": args.max_current,
                        "center_deg": center,
                        "poll_rate_achieved_hz": poller.achieved_hz,
                        "samples_recorded": len(poller.samples),
                    },
                    "gripper_calibration": (
                        {
                            "csv_path": str(args.gripper_calibration_csv),
                            "num_points": len(gripper_calib[0]),
                            # The fixed offset applied to the CSV's own (relative)
                            # servo_position_deg column -- see
                            # --gripper-calibration-offset-deg / GRIPPER_CALIBRATION_BIAS_DEG --
                            # recorded here so a run can be audited even if the
                            # default offset is later re-measured.
                            "offset_deg": args.gripper_calibration_offset_deg,
                            "max_width_mm": args.gripper_max_width_mm,
                            "position_range_deg": [float(gripper_calib[0].min()), float(gripper_calib[0].max())],
                            "width_range_mm": [float(gripper_calib[1].min()), float(gripper_calib[1].max())],
                        }
                        if gripper_calib is not None
                        else None
                    ),
                    "color_fps_requested": args.color_fps,
                    "orbslam": (
                        {
                            "settings_path": str(args.orbslam_settings),
                            "vocab_path": str(args.orbslam_vocab),
                            "frames_tracked": sum(1 for r in orb_trajectory if r["is_lost"] == 0),
                            "frames_total": len(orb_trajectory),
                            "trajectory_path": "camera_trajectory.csv",
                            # See write_camera_trajectory_csv's map_epoch docstring: a map
                            # reset means the trajectory has >1 coordinate frame spliced
                            # together under one file. 0 resets = camera_trajectory.csv
                            # is one continuous coordinate frame throughout.
                            "map_resets": orb_map_resets[0],
                            "map_segments": orb_map_resets[0] + 1,
                        }
                        if orb_tracker is not None
                        else None
                    ),
                    "openvins": (
                        {
                            "config_path": str(args.openvins_config),
                            "frames_tracked": sum(1 for r in ov_trajectory if r["is_lost"] == 0),
                            "frames_total": len(ov_trajectory),
                            "trajectory_path": (
                                "camera_trajectory_openvins.csv" if orb_tracker is not None else "camera_trajectory.csv"
                            ),
                        }
                        if ov_tracker is not None
                        else None
                    ),
                }
            )
            cam.write_json(metadata_path, metadata)

            print(f"Wrote: {scan_dir / 'servo.csv'}")
            print(f"Wrote: {scan_dir / 'synced.csv'}  (one row per camera frame, all streams joined)")
            print(f"Metadata: {metadata_path}")

            episode_results.append({"scan_dir": scan_dir, "complete": bool(capture_result.get("complete"))})

            if abort_batch_event.is_set():
                print(f"\nBatch stopped after episode {episode_idx}/{args.episodes}.")
                break

        if monitor is not None:
            monitor.close()
        poller.stop()

        if args.episodes > 1 or not episode_results:
            completed = sum(1 for r in episode_results if r["complete"])
            print(f"\nBatch summary: {len(episode_results)}/{args.episodes} episodes recorded "
                  f"({completed} complete, {len(episode_results) - completed} incomplete).")
            for r in episode_results:
                print(f"  {'OK  ' if r['complete'] else 'WARN'} {r['scan_dir']}")

        return 0 if episode_results and all(r["complete"] for r in episode_results) else 1

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if poller is not None:
            poller.stop()
        if monitor is not None:
            monitor.close()
        if orb_worker is not None:
            orb_worker.close()
        if orb_tracker is not None:
            orb_tracker.close()
        if ov_tracker is not None:
            ov_tracker.close()
        servo.close()
        print("Servo torque disabled, port closed.")


if __name__ == "__main__":
    raise SystemExit(main())
