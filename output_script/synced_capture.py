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

    WINDOW_NAME = "Synced capture -- camera + servo position (press q or close to stop)"

    # Panel look: white background, blue line, black text -- easy to change
    # here if the palette should change again.
    BG_COLOR = (255, 255, 255)     # BGR: white
    LINE_COLOR = (180, 119, 31)    # BGR: blue
    TEXT_COLOR = (20, 20, 20)      # BGR: near-black
    GRID_COLOR = (215, 215, 215)   # BGR: light gray
    GRID_DIVISIONS = 4              # grid cells per axis

    def __init__(
        self,
        cv2_module: Any,
        window_seconds: float,
        monitor_fps: float,
        on_stop_requested: "Any",
        plot_width: int = 420,
    ):
        self.cv2 = cv2_module
        self.window_seconds = window_seconds
        self.min_interval_ns = int(1e9 / monitor_fps) if monitor_fps > 0 else 0
        self.plot_width = plot_width
        self.on_stop_requested = on_stop_requested
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

        cv2.putText(panel, "Servo position (deg)", (margin, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.TEXT_COLOR, 1, cv2.LINE_AA)

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
        cv2.putText(panel, f"{self.total_samples_seen} pts, last={last_str}", (margin, height - 14),
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

        cv2.imshow(self.WINDOW_NAME, combined)
        key = cv2.waitKey(1) & 0xFF
        window_closed = cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        if key in (27, ord("q")) or window_closed:
            self._stop_event.set()
            if self.on_stop_requested is not None:
                self.on_stop_requested()

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


def write_servo_csv(path: Path, matched_rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "frame_id",
        "frame_host_time_ns",
        "servo_host_time_ns",
        "sync_error_ms",
        "position_deg",
        "velocity_deg_s",
        "current_ma",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
        "servo_position_deg",
        "servo_velocity_deg_s",
        "servo_current_ma",
        "servo_sync_error_ms",
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
                    "servo_position_deg": servo["position_deg"] if servo else None,
                    "servo_velocity_deg_s": servo["velocity_deg_s"] if servo else None,
                    "servo_current_ma": servo["current_ma"] if servo else None,
                    "servo_sync_error_ms": servo["sync_error_ms"] if servo else None,
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

    # Monitor -- one cv2 window: camera frame + servo position plot
    # composited side by side. See CombinedMonitor's docstring.
    parser.add_argument("--monitor", dest="monitor", action=argparse.BooleanOptionalAction, default=True,
                         help="Show the live camera+servo-position window while recording (default: on). "
                              "Press 'q' or close the window to stop recording early.")
    parser.add_argument("--monitor-fps", type=float, default=15.0, dest="monitor_fps",
                         help="Monitor window redraw rate, decoupled from the recording rate (default 15Hz).")
    parser.add_argument("--monitor-window", type=float, default=10.0, dest="monitor_window",
                         help="Seconds of servo-position history shown in the live plot at once.")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rs = cam.import_realsense()

    port = servo_mod.resolve_port(args.servo_port)
    servo = servo_mod.DynamixelPositionSpring(port, args.servo_baud, args.servo_id)
    # Declared before the try block (not just inside it) so `finally` can
    # always safely check them, even if something fails before either one
    # gets constructed.
    poller: ServoPoller | None = None
    monitor: CombinedMonitor | None = None

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
        scan_dir = cam.next_scan_dir(args.data_dir)
        scan_dir.mkdir()

        requested_color = cam.optional_profile(args.color_width, args.color_height, args.color_fps, "color")

        capture = cam.RealSenseCapture(
            rs=rs,
            scan_dir=scan_dir,
            max_duration_seconds=args.max_duration,
            requested_color=requested_color,
            requested_depth=None,
            enable_imu=True,
            record_rgb=True,
            record_depth=args.depth_flag,
            queue_size=args.queue_size,
            imu_fps=args.imu_fps,
            # Not using RealSenseCapture's own cv2 preview window -- Combined
            # Monitor builds its own single window (camera + servo plot)
            # instead, fed via on_preview_frame below. preview_fps still
            # throttles how often on_preview_frame fires (see _update_preview).
            show_preview=False,
            preview_fps=args.monitor_fps,
        )

        poller = ServoPoller(servo, capture.active)
        poller.start()

        if args.monitor:
            try:
                cv2 = cam.import_opencv()

                def _on_monitor_closed() -> None:
                    print("\nMonitor window closed -- stopping.")
                    capture.stop_event.set()

                monitor = CombinedMonitor(
                    cv2,
                    window_seconds=args.monitor_window,
                    monitor_fps=args.monitor_fps,
                    on_stop_requested=_on_monitor_closed,
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
        cam.write_calibration(rs, capture.profile, scan_dir / "calibration.json", args.depth_flag)

        print(f"\nOutput folder: {scan_dir}")
        print(f"Servo poll thread running (target: as fast as the USB link allows).")
        input("Press Enter to start capture...")

        def _wait_for_stop() -> None:
            input("Recording... press Enter to stop.\n")
            capture.stop_event.set()

        stop_thread = threading.Thread(target=_wait_for_stop, daemon=True)
        stop_thread.start()

        capture_result = capture.capture()

        if monitor is not None:
            monitor.close()

        poller.stop()
        if poller.error is not None:
            print(f"\nWarning: servo poller stopped early due to an error: {poller.error}")

        gyro_matched = cam.match_samples_to_frames(capture.frame_rows, capture.gyro_rows)
        accel_matched = cam.match_samples_to_frames(capture.frame_rows, capture.accel_rows)
        servo_matched = match_servo_to_frames(capture.frame_rows, poller.samples)

        write_servo_csv(scan_dir / "servo.csv", servo_matched)
        write_synced_csv(scan_dir / "synced.csv", capture.frame_rows, gyro_matched, accel_matched, servo_matched)

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
                "color_fps_requested": args.color_fps,
            }
        )
        cam.write_json(metadata_path, metadata)

        print(f"\nWrote: {scan_dir / 'servo.csv'}")
        print(f"Wrote: {scan_dir / 'synced.csv'}  (one row per camera frame, all streams joined)")
        print(f"Metadata: {metadata_path}")

        return 0 if capture_result.get("complete") else 1

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if poller is not None:
            poller.stop()
        if monitor is not None:
            monitor.close()
        servo.close()
        print("Servo torque disabled, port closed.")


if __name__ == "__main__":
    raise SystemExit(main())
