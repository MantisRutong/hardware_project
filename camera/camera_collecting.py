#!/usr/bin/env python3
"""Capture a RealSense D435i RGB-D-IMU scan with manual start/stop.

The script creates enumerated scan folders under the configured data directory,
waits for you to press Enter to start, records RGB images, aligned depth images,
and IMU samples (gyro/accel) matched one-to-one with each saved video frame by
nearest timestamp, then stops when you press Enter again (or Ctrl+C).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image


DEFAULT_DATA_DIR = Path("/home/core/Desktop/recording")

COLOR_PROFILES = (
    # 60fps first: this is the default frequency for this rig. Confirmed
    # working at 848x480 (scan_0001-0047, 2026-07-31); the D435i's color
    # sensor doesn't support 60fps at 1280x720/1920x1080, so those stay in
    # the 30fps fallback list below in case 60fps negotiation ever fails.
    (848, 480, 60),
    (640, 480, 60),
    (1920, 1080, 30),
    (1280, 720, 30),
    (960, 540, 30),
    (848, 480, 30),
    (640, 480, 30),
)
DEPTH_PROFILES = (
    (1280, 720, 30),
    (848, 480, 30),
    (640, 480, 30),
)


@dataclass(frozen=True)
class VideoProfile:
    width: int
    height: int
    fps: int


@dataclass
class CaptureCounters:
    rgbd_saved: int = 0
    video_frames_dropped: int = 0
    incomplete_frames: int = 0
    gyro_samples: int = 0
    accel_samples: int = 0


class RealSenseCapture:
    def __init__(
        self,
        rs: Any,
        scan_dir: Path,
        max_duration_seconds: float | None,
        requested_color: VideoProfile | None,
        requested_depth: VideoProfile | None,
        enable_imu: bool,
        record_rgb: bool,
        record_depth: bool,
        queue_size: int,
        imu_fps: int,
        show_preview: bool,
        preview_fps: float,
        on_frame: Callable[[int, int, Any], None] | None = None,
        on_preview_frame: Callable[[Any], None] | None = None,
    ) -> None:
        self.rs = rs
        self.scan_dir = scan_dir
        # Optional hook, called synchronously (on the same thread as
        # capture()'s main loop, once per saved frame) right after a frame
        # is written: on_frame(frame_id, host_time_ns, color_ndarray_or_None).
        # Lets an external caller (e.g. a combined camera+servo recorder)
        # sync other sensors to each frame and/or drive a live view, without
        # this class needing to know anything about what's on the other end.
        # A raised exception here is swallowed (see _save_frameset) so a
        # flaky monitor/callback can never take down the actual recording.
        self.on_frame = on_frame
        # Optional hook, called from _update_preview with a fresh RGB
        # ndarray -- UNLIKE on_frame, this fires regardless of recording
        # state (see _frame_callback), so a live view can show real camera
        # content before recording starts and after it stops, not only
        # during the active window. Setting this also turns on the
        # underlying preview machinery even if show_preview is False (i.e.
        # you can get frames here without RealSenseCapture opening its own
        # cv2 window).
        self.on_preview_frame = on_preview_frame
        self.max_duration_seconds = max_duration_seconds
        self.requested_color = requested_color
        self.requested_depth = requested_depth
        self.enable_imu = enable_imu
        self.record_rgb = record_rgb
        self.record_depth = record_depth
        self.queue_size = queue_size
        self.imu_fps = imu_fps
        self.show_preview = show_preview
        self.preview_interval_ns = int(1e9 / preview_fps) if preview_fps > 0 else 0
        self.latest_preview_frame: Any | None = None
        self._last_preview_ns = 0
        self.video_queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self.imu_lock = threading.Lock()
        self.gyro_rows: list[dict[str, Any]] = []
        self.accel_rows: list[dict[str, Any]] = []
        self.counters = CaptureCounters()
        self.pipeline: Any | None = None
        self.profile: Any | None = None
        self.align: Any | None = None
        self.syncer: Any | None = None
        self.active = threading.Event()
        self.stop_event = threading.Event()
        self.end_mono_ns: int | None = None
        self.start_mono_ns: int | None = None

    def start_camera(self) -> None:
        attempts: list[tuple[VideoProfile, VideoProfile | None, bool]] = []

        color_profiles = (
            (self.requested_color,)
            if self.requested_color is not None
            else tuple(VideoProfile(*p) for p in COLOR_PROFILES)
        )
        depth_profiles = (
            (self.requested_depth,)
            if self.requested_depth is not None
            else tuple(VideoProfile(*p) for p in DEPTH_PROFILES)
        )

        if self.record_depth:
            for color_profile in color_profiles:
                for depth_profile in depth_profiles:
                    attempts.append((color_profile, depth_profile, self.enable_imu))
            if self.enable_imu:
                for color_profile in color_profiles:
                    for depth_profile in depth_profiles:
                        attempts.append((color_profile, depth_profile, False))
        else:
            # Depth is not being saved, so skip the depth stream entirely: a
            # depth sensor that stalls can otherwise block color framesets too,
            # since the pipeline waits for both streams to sync.
            for color_profile in color_profiles:
                attempts.append((color_profile, None, self.enable_imu))
            if self.enable_imu:
                for color_profile in color_profiles:
                    attempts.append((color_profile, None, False))

        errors: list[str] = []
        for color_profile, depth_profile, with_imu in attempts:
            self.pipeline = self.rs.pipeline()
            config = self.rs.config()
            config.enable_stream(
                self.rs.stream.color,
                color_profile.width,
                color_profile.height,
                self.rs.format.rgb8,
                color_profile.fps,
            )
            if depth_profile is not None:
                config.enable_stream(
                    self.rs.stream.depth,
                    depth_profile.width,
                    depth_profile.height,
                    self.rs.format.z16,
                    depth_profile.fps,
                )
            if with_imu:
                config.enable_stream(self.rs.stream.gyro, self.rs.format.motion_xyz32f, self.imu_fps)
                config.enable_stream(self.rs.stream.accel, self.rs.format.motion_xyz32f, self.imu_fps)

            try:
                self.syncer = self.rs.syncer(self.queue_size)
                self.profile = self.pipeline.start(config, self._frame_callback)
                self.align = (
                    self.rs.align(self.rs.stream.color) if depth_profile is not None else None
                )
                self.enable_imu = with_imu
                return
            except Exception as exc:  # noqa: BLE001 - pyrealsense throws runtime errors
                errors.append(
                    f"color={color_profile} depth={depth_profile} imu={with_imu}: {exc}"
                )
                try:
                    self.pipeline.stop()
                except Exception:
                    pass

        detail = "\n".join(errors[-6:])
        raise RuntimeError(f"Unable to start RealSense camera. Recent errors:\n{detail}")

    def stop_camera(self) -> None:
        self.active.clear()
        if self.pipeline is not None:
            self.pipeline.stop()

    def print_camera_summary(self) -> None:
        if self.profile is None:
            raise RuntimeError("Camera is not started")

        device = self.profile.get_device()
        print("RealSense camera ready")
        print(f"  Device: {safe_device_name(self.rs, device)}")
        print(f"  Serial: {safe_device_info(self.rs, device, self.rs.camera_info.serial_number)}")
        print(f"  Firmware: {safe_device_info(self.rs, device, self.rs.camera_info.firmware_version)}")
        print(f"  IMU enabled: {self.enable_imu}")

        streams = (
            (self.rs.stream.color, self.rs.stream.depth)
            if self.record_depth
            else (self.rs.stream.color,)
        )
        for stream in streams:
            stream_profile = self.profile.get_stream(stream)
            video_profile = stream_profile.as_video_stream_profile()
            intr = video_profile.get_intrinsics()
            print(
                "  "
                f"{stream}: {intr.width}x{intr.height} "
                f"fx={intr.fx:.3f} fy={intr.fy:.3f} "
                f"cx={intr.ppx:.3f} cy={intr.ppy:.3f}"
            )

        if self.record_depth:
            depth_scale = get_depth_scale(device)
            print(f"  Depth scale: {depth_scale}")

    def capture(self) -> dict[str, Any]:
        if self.profile is None:
            raise RuntimeError("Camera is not started")

        rgb_dir = self.scan_dir / "rgb"
        depth_dir = self.scan_dir / "depth"
        imu_dir = self.scan_dir / "imu"
        if self.record_rgb:
            rgb_dir.mkdir(parents=True, exist_ok=True)
        if self.record_depth:
            depth_dir.mkdir(parents=True, exist_ok=True)
        if self.enable_imu:
            imu_dir.mkdir(parents=True, exist_ok=True)

        self.start_mono_ns = time.monotonic_ns()
        self.end_mono_ns = (
            self.start_mono_ns + int(self.max_duration_seconds * 1e9)
            if self.max_duration_seconds is not None
            else None
        )
        started_at_wall = now_iso()
        self.active.set()

        # Instance attribute (not just a local) so external code -- e.g. a
        # combined recorder that also logs a second sensor -- can read the
        # per-frame timestamps/paths back out after capture() returns, to
        # match its own samples to each frame the same way IMU rows already
        # are below.
        self.frame_rows: list[dict[str, Any]] = []
        frame_rows = self.frame_rows
        last_progress_second = -10
        interrupted = False

        cv2 = import_opencv() if self.show_preview else None
        preview_window = "RealSense preview (press q or close window to stop)"
        if cv2 is not None:
            cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)

        try:
            while not self.stop_event.is_set():
                if self.end_mono_ns is not None and time.monotonic_ns() >= self.end_mono_ns:
                    break
                self._drain_video_queue(
                    rgb_dir,
                    depth_dir,
                    frame_rows,
                    block_timeout=0.2,
                    max_items=1,
                )
                if cv2 is not None:
                    if self.latest_preview_frame is not None:
                        cv2.imshow(preview_window, self.latest_preview_frame)
                    key = cv2.waitKey(1) & 0xFF
                    window_closed = cv2.getWindowProperty(preview_window, cv2.WND_PROP_VISIBLE) < 1
                    if key in (27, ord("q")) or window_closed:
                        self.stop_event.set()
                elapsed = (time.monotonic_ns() - self.start_mono_ns) / 1e9
                progress_second = int(elapsed)
                if progress_second - last_progress_second >= 10:
                    last_progress_second = progress_second
                    print(
                        f"  {progress_second:3d}s elapsed | "
                        f"RGB-D {self.counters.rgbd_saved} | "
                        f"gyro {self.counters.gyro_samples} | "
                        f"accel {self.counters.accel_samples} | "
                        f"dropped {self.counters.video_frames_dropped}"
                    )
        except KeyboardInterrupt:
            interrupted = True
            print("\nCapture interrupted; writing metadata for partial scan.")
        finally:
            self.active.clear()
            if cv2 is not None:
                cv2.destroyWindow(preview_window)

        capture_end_mono_ns = (
            min(time.monotonic_ns(), self.end_mono_ns)
            if self.end_mono_ns is not None
            else time.monotonic_ns()
        )
        capture_ended_at_wall = now_iso()
        print("Collection complete. You can stop moving/holding the camera; saving files now.")
        self._drain_video_queue(rgb_dir, depth_dir, frame_rows, block_timeout=0.0)
        processing_ended_at_wall = now_iso()

        self._write_frame_indexes(frame_rows)
        if self.enable_imu:
            self._write_imu_csvs(imu_dir, frame_rows)

        complete = not interrupted and self.counters.rgbd_saved > 0
        result = {
            "started_at": started_at_wall,
            "ended_at": capture_ended_at_wall,
            "processing_ended_at": processing_ended_at_wall,
            "complete": complete,
            "max_duration_seconds": self.max_duration_seconds,
            "actual_capture_seconds": (
                (capture_end_mono_ns - self.start_mono_ns) / 1e9
                if self.start_mono_ns is not None
                else 0.0
            ),
            "processing_seconds_after_capture": (
                (time.monotonic_ns() - capture_end_mono_ns) / 1e9
            ),
            "counts": {
                "rgbd_frames": self.counters.rgbd_saved,
                "video_frames_dropped": self.counters.video_frames_dropped,
                "incomplete_frames": self.counters.incomplete_frames,
                "gyro_samples": self.counters.gyro_samples,
                "accel_samples": self.counters.accel_samples,
            },
        }
        if self.counters.rgbd_saved == 0:
            result["warning"] = "No RGB-D frames were saved."
        return result

    def _frame_callback(self, frame: Any) -> None:
        active = self.active.is_set()
        preview_wanted = self.show_preview or self.on_preview_frame is not None
        if not active and not preview_wanted:
            # Nothing wants this frame -- skip the SDK calls below entirely
            # (frameset conversion + color/depth readiness checks aren't
            # free) rather than doing them just to throw the result away.
            return

        host_time_ns = time.time_ns()
        frameset = as_frameset(frame) if is_frameset(frame) else None
        frameset_ready = frameset is not None and (
            frameset_has_color_and_depth(self.rs, frameset)
            if self.record_depth
            else frameset_has_color(self.rs, frameset)
        )

        # Preview runs regardless of recording state (self.active) -- lets a
        # live view (the built-in cv2 window, or an external on_preview_frame
        # consumer) show real content before recording starts and after it
        # stops, not just during the active window. Actual recording
        # (enqueuing frames to save, logging IMU) still requires
        # self.active, checked below -- this only affects what's DISPLAYED,
        # never what's WRITTEN to disk.
        if frameset_ready and preview_wanted:
            self._update_preview(frameset)

        if not active:
            return
        if self.end_mono_ns is not None and time.monotonic_ns() > self.end_mono_ns:
            return

        if is_motion_frame(frame):
            self._record_motion_frame(frame, host_time_ns)
            return

        if frameset is not None and frameset_ready:
            self._enqueue_video_frameset(frameset, host_time_ns)
        else:
            self._sync_video_frame(frame)

    def _update_preview(self, frameset: Any) -> None:
        now_ns = time.monotonic_ns()
        if now_ns - self._last_preview_ns < self.preview_interval_ns:
            return
        try:
            color_frame = frameset.get_color_frame()
            if not color_frame:
                return
            color = np.asanyarray(color_frame.get_data())
        except Exception:
            return
        self.latest_preview_frame = color[:, :, ::-1]  # RGB -> BGR for cv2
        self._last_preview_ns = now_ns
        self._call_hook(self.on_preview_frame, color)  # RGB, same contract as on_frame

    @staticmethod
    def _call_hook(hook: Callable[..., None] | None, *args: Any) -> None:
        """Call an optional external hook (on_frame / on_preview_frame),
        swallowing any exception it raises -- a broken/slow consumer (e.g. a
        live monitor redraw) must never be able to affect capture."""
        if hook is not None:
            try:
                hook(*args)
            except Exception:
                pass

    def _record_motion_frame(self, frame: Any, host_time_ns: int) -> None:
        motion_frame = frame.as_motion_frame()
        motion = motion_frame.get_motion_data()
        profile = motion_frame.get_profile()
        stream_type = profile.stream_type()
        row = {
            "frame_number": int(motion_frame.get_frame_number()),
            "timestamp_seconds": float(motion_frame.get_timestamp()) / 1000.0,
            "rs_timestamp_ms": float(motion_frame.get_timestamp()),
            "timestamp_domain": str(motion_frame.get_frame_timestamp_domain()),
            "host_time_ns": host_time_ns,
            "x": float(motion.x),
            "y": float(motion.y),
            "z": float(motion.z),
        }

        with self.imu_lock:
            if stream_type == self.rs.stream.gyro:
                self.gyro_rows.append(row)
                self.counters.gyro_samples += 1
            elif stream_type == self.rs.stream.accel:
                self.accel_rows.append(row)
                self.counters.accel_samples += 1

    def _drain_video_queue(
        self,
        rgb_dir: Path,
        depth_dir: Path,
        frame_rows: list[dict[str, Any]],
        block_timeout: float,
        max_items: int | None = None,
    ) -> None:
        processed = 0
        while True:
            if max_items is not None and processed >= max_items:
                return
            try:
                frameset, host_time_ns = self.video_queue.get(timeout=block_timeout)
            except queue.Empty:
                synced = self._next_synced_frameset(block_timeout)
                if synced is None:
                    return
                frameset, host_time_ns = synced
            block_timeout = 0.0
            self._save_frameset(frameset, host_time_ns, rgb_dir, depth_dir, frame_rows)
            processed += 1

    def _next_synced_frameset(self, block_timeout: float) -> tuple[Any, int] | None:
        if self.syncer is None:
            return None

        timeout_ms = max(1, int(block_timeout * 1000))
        try:
            frameset = self.syncer.wait_for_frames(timeout_ms)
        except Exception:
            return None

        try:
            frameset.keep()
        except Exception:
            pass
        return frameset, time.time_ns()

    def _enqueue_video_frameset(self, frameset: Any, host_time_ns: int) -> None:
        try:
            frameset.keep()
        except Exception:
            pass
        try:
            self.video_queue.put_nowait((frameset, host_time_ns))
        except queue.Full:
            self.counters.video_frames_dropped += 1

    def _sync_video_frame(self, frame: Any) -> None:
        if self.syncer is None:
            return
        try:
            self.syncer(frame)
        except Exception:
            pass

    def _save_frameset(
        self,
        frameset: Any,
        host_time_ns: int,
        rgb_dir: Path,
        depth_dir: Path,
        frame_rows: list[dict[str, Any]],
    ) -> None:
        if self.align is not None:
            try:
                aligned = self.align.process(frameset)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
            except Exception:
                self.counters.incomplete_frames += 1
                return

            if not color_frame or not depth_frame:
                self.counters.incomplete_frames += 1
                return
        else:
            color_frame = frameset.get_color_frame()
            depth_frame = None
            if not color_frame:
                self.counters.incomplete_frames += 1
                return

        frame_id = self.counters.rgbd_saved
        color_ts_s = float(color_frame.get_timestamp()) / 1000.0
        depth_ts_s = float(depth_frame.get_timestamp()) / 1000.0 if depth_frame is not None else None
        color_ts_ns = int(round(color_ts_s * 1e9))
        depth_ts_ns = int(round(depth_ts_s * 1e9)) if depth_ts_s is not None else None

        rgb_name = f"{frame_id:06d}_{color_ts_ns}.png"
        rgb_rel = f"rgb/{rgb_name}" if self.record_rgb else None
        depth_rel = None
        if self.record_depth and depth_frame is not None:
            depth_name = f"{frame_id:06d}_{depth_ts_ns}.png"
            depth_rel = f"depth/{depth_name}"

        if self.record_rgb:
            color = np.asanyarray(color_frame.get_data())
            Image.fromarray(color, mode="RGB").save(rgb_dir / rgb_name, compress_level=1)
        if self.record_depth and depth_frame is not None:
            depth = np.asanyarray(depth_frame.get_data())
            Image.fromarray(depth).save(depth_dir / depth_name, compress_level=1)

        frame_rows.append(
            {
                "frame_id": frame_id,
                "color_timestamp_seconds": color_ts_s,
                "depth_timestamp_seconds": depth_ts_s,
                "color_rs_timestamp_ms": float(color_frame.get_timestamp()),
                "depth_rs_timestamp_ms": (
                    float(depth_frame.get_timestamp()) if depth_frame is not None else None
                ),
                "color_timestamp_domain": str(color_frame.get_frame_timestamp_domain()),
                "depth_timestamp_domain": (
                    str(depth_frame.get_frame_timestamp_domain()) if depth_frame is not None else None
                ),
                "host_time_ns": host_time_ns,
                "rgb_path": rgb_rel,
                "depth_path": depth_rel,
                "color_frame_number": int(color_frame.get_frame_number()),
                "depth_frame_number": (
                    int(depth_frame.get_frame_number()) if depth_frame is not None else None
                ),
            }
        )
        self.counters.rgbd_saved += 1
        self._call_hook(self.on_frame, frame_id, host_time_ns, color if self.record_rgb else None)

    def _write_frame_indexes(self, frame_rows: list[dict[str, Any]]) -> None:
        with (self.scan_dir / "frames.csv").open("w", newline="") as f:
            fieldnames = [
                "frame_id",
                "color_timestamp_seconds",
                "depth_timestamp_seconds",
                "color_rs_timestamp_ms",
                "depth_rs_timestamp_ms",
                "color_timestamp_domain",
                "depth_timestamp_domain",
                "host_time_ns",
                "rgb_path",
                "depth_path",
                "color_frame_number",
                "depth_frame_number",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(frame_rows)

        if self.record_rgb:
            with (self.scan_dir / "rgb.txt").open("w") as f_rgb:
                f_rgb.write("# timestamp rgb_path\n")
                for row in frame_rows:
                    f_rgb.write(f"{row['color_timestamp_seconds']:.9f} {row['rgb_path']}\n")

        if self.record_depth:
            with (self.scan_dir / "depth.txt").open("w") as f_depth:
                f_depth.write("# timestamp depth_path\n")
                for row in frame_rows:
                    f_depth.write(
                        f"{row['depth_timestamp_seconds']:.9f} {row['depth_path']}\n"
                    )

    def _write_imu_csvs(self, imu_dir: Path, frame_rows: list[dict[str, Any]]) -> None:
        fieldnames = [
            "frame_id",
            "frame_timestamp_seconds",
            "frame_number",
            "timestamp_seconds",
            "rs_timestamp_ms",
            "timestamp_domain",
            "host_time_ns",
            "x",
            "y",
            "z",
        ]
        with self.imu_lock:
            gyro_rows = list(self.gyro_rows)
            accel_rows = list(self.accel_rows)

        for filename, rows in (("gyro.csv", gyro_rows), ("accel.csv", accel_rows)):
            matched_rows = match_samples_to_frames(frame_rows, rows)
            with (imu_dir / filename).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(matched_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a RealSense D435i RGB-D-IMU scan."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help=(
            "Optional safety cap in seconds; capture auto-stops if reached. "
            "By default there is no cap and capture runs until you press Enter again."
        ),
    )
    parser.add_argument("--queue-size", type=int, default=256)
    parser.add_argument(
        "--rgb-flag",
        dest="rgb_flag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save RGB images to disk (default: on). Use --no-rgb-flag to skip.",
    )
    parser.add_argument(
        "--depth-flag",
        dest="depth_flag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save depth images to disk (default: on). Use --no-depth-flag to skip.",
    )
    parser.add_argument(
        "--imu-flag",
        dest="imu_flag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable and record IMU gyro/accel samples (default: on). Use --no-imu-flag to skip.",
    )
    parser.add_argument("--color-width", type=int)
    parser.add_argument("--color-height", type=int)
    parser.add_argument("--color-fps", type=int, default=60)
    parser.add_argument("--depth-width", type=int)
    parser.add_argument("--depth-height", type=int)
    parser.add_argument("--depth-fps", type=int, default=30)
    parser.add_argument(
        "--imu-fps",
        type=int,
        default=200,
        help=(
            "Sample rate in Hz for both gyro and accel, so the two IMU channels "
            "stay synchronized at the same rate (default: 200). Supported: "
            "100/200/400 for accel, 200/400 for gyro."
        ),
    )
    parser.add_argument(
        "--preview",
        dest="preview",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show a live color preview window while recording (default: off).",
    )
    parser.add_argument(
        "--preview-fps",
        type=float,
        default=15.0,
        help=(
            "Refresh rate of the --preview window in Hz (default: 15). "
            "Decoupled from --color-fps so a slower preview doesn't need to "
            "match the recorded frame rate."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rs = import_realsense()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    scan_dir = next_scan_dir(args.data_dir)
    scan_dir.mkdir()

    requested_color = optional_profile(
        args.color_width, args.color_height, args.color_fps, "color"
    )
    requested_depth = optional_profile(
        args.depth_width, args.depth_height, args.depth_fps, "depth"
    )

    capture = RealSenseCapture(
        rs=rs,
        scan_dir=scan_dir,
        max_duration_seconds=args.max_duration,
        requested_color=requested_color,
        requested_depth=requested_depth,
        enable_imu=args.imu_flag,
        record_rgb=args.rgb_flag,
        record_depth=args.depth_flag,
        queue_size=args.queue_size,
        imu_fps=args.imu_fps,
        show_preview=args.preview,
        preview_fps=args.preview_fps,
    )

    metadata: dict[str, Any] = {
        "script": Path(__file__).name,
        "script_version": 1,
        "scan_name": scan_dir.name,
        "scan_dir": str(scan_dir),
        "created_at": now_iso(),
        "max_duration_seconds": args.max_duration,
        "rgb_flag": args.rgb_flag,
        "depth_flag": args.depth_flag,
        "imu_flag": args.imu_flag,
        "imu_fps": args.imu_fps,
    }

    try:
        capture.start_camera()
        capture.print_camera_summary()
        write_calibration(rs, capture.profile, scan_dir / "calibration.json", args.depth_flag)

        print(f"\nOutput folder: {scan_dir}")
        input("Press Enter to start capture...")

        def _wait_for_stop() -> None:
            input(
                "Recording... press Enter to stop"
                + (", or press q / close the preview window.\n" if args.preview else ".\n")
            )
            capture.stop_event.set()

        # The stop prompt runs on a background thread so the main thread is
        # free to run capture()'s loop, which pumps the cv2 preview window
        # when --preview is on (GUI event handling needs a consistent thread).
        stop_thread = threading.Thread(target=_wait_for_stop, daemon=True)
        stop_thread.start()
        capture_result = capture.capture()

        metadata.update(capture_result)
    except Exception as exc:  # noqa: BLE001 - produce metadata before exiting
        metadata["complete"] = False
        metadata["error"] = str(exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return_code = 1
    else:
        return_code = 0 if metadata.get("complete") else 1
    finally:
        try:
            capture.stop_camera()
        except Exception:
            pass
        metadata["finished_at"] = now_iso()
        metadata["imu_enabled"] = capture.enable_imu
        metadata["counts"] = metadata.get(
            "counts",
            {
                "rgbd_frames": capture.counters.rgbd_saved,
                "video_frames_dropped": capture.counters.video_frames_dropped,
                "incomplete_frames": capture.counters.incomplete_frames,
                "gyro_samples": capture.counters.gyro_samples,
                "accel_samples": capture.counters.accel_samples,
            },
        )
        write_json(scan_dir / "metadata.json", metadata)

    print(f"Metadata written to {scan_dir / 'metadata.json'}")
    return return_code


def import_realsense() -> Any:
    try:
        import pyrealsense2 as rs  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "pyrealsense2 is not installed in this Python environment.\n"
            "Install it in the environment you will use for capture, for example:\n"
            "  pip install pyrealsense2"
        ) from exc
    return rs


def import_opencv() -> Any:
    # The bundled Qt in opencv-python has no Wayland plugin; force the X11
    # (xcb) backend, which works via XWayland on Wayland sessions too.
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "opencv-python is not installed in this Python environment.\n"
            "Install it to use --preview, for example:\n"
            "  pip install opencv-python"
        ) from exc
    return cv2


def optional_profile(
    width: int | None, height: int | None, fps: int, label: str
) -> VideoProfile | None:
    if width is None and height is None:
        return None
    if width is None or height is None:
        raise SystemExit(f"Specify both --{label}-width and --{label}-height.")
    return VideoProfile(width=width, height=height, fps=fps)


def next_scan_dir(data_dir: Path) -> Path:
    max_index = 0
    for child in data_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("scan_"):
            continue
        suffix = child.name.removeprefix("scan_")
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return data_dir / f"scan_{max_index + 1:04d}"


def write_calibration(rs: Any, profile: Any, path: Path, include_depth: bool) -> None:
    device = profile.get_device()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()

    calibration: dict[str, Any] = {
        "device": device_info(rs, device),
        "color": video_stream_info(color_profile),
        "depth_aligned_to_color": include_depth,
    }

    if include_depth:
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        calibration["depth_scale"] = get_depth_scale(device)
        calibration["depth"] = video_stream_info(depth_profile)
        try:
            calibration["extrinsics"] = {
                "depth_to_color": extrinsics_to_dict(depth_profile.get_extrinsics_to(color_profile)),
                "color_to_depth": extrinsics_to_dict(color_profile.get_extrinsics_to(depth_profile)),
            }
        except Exception as exc:
            calibration["extrinsics_error"] = str(exc)

    write_json(path, calibration)


def video_stream_info(video_profile: Any) -> dict[str, Any]:
    intr = video_profile.get_intrinsics()
    return {
        "stream_name": call_profile_method(video_profile, "stream_name"),
        "stream_type": str(call_profile_method(video_profile, "stream_type")),
        "format": str(call_profile_method(video_profile, "format")),
        "fps": call_profile_method(video_profile, "fps"),
        "intrinsics": {
            "width": int(intr.width),
            "height": int(intr.height),
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "cx": float(intr.ppx),
            "cy": float(intr.ppy),
            "distortion_model": str(intr.model),
            "coeffs": [float(x) for x in intr.coeffs],
        },
    }


def call_profile_method(profile: Any, method_name: str) -> Any:
    try:
        value = getattr(profile, method_name)()
    except Exception:
        return None
    if method_name == "fps" and value is not None:
        return int(value)
    return value


def extrinsics_to_dict(extrinsics: Any) -> dict[str, Any]:
    return {
        "rotation_row_major": [float(x) for x in extrinsics.rotation],
        "translation": [float(x) for x in extrinsics.translation],
    }


def device_info(rs: Any, device: Any) -> dict[str, Any]:
    fields = {
        "name": rs.camera_info.name,
        "serial_number": rs.camera_info.serial_number,
        "firmware_version": rs.camera_info.firmware_version,
        "physical_port": rs.camera_info.physical_port,
        "product_id": rs.camera_info.product_id,
        "product_line": rs.camera_info.product_line,
    }
    return {key: safe_device_info(rs, device, value) for key, value in fields.items()}


def safe_device_name(rs: Any, device: Any) -> str:
    return safe_device_info(rs, device, rs.camera_info.name)


def safe_device_info(rs: Any, device: Any, field: Any) -> str | None:
    try:
        if device.supports(field):
            return str(device.get_info(field))
    except Exception:
        return None
    return None


def get_depth_scale(device: Any) -> float | None:
    try:
        return float(device.first_depth_sensor().get_depth_scale())
    except Exception:
        return None


def is_frameset(frame: Any) -> bool:
    try:
        return bool(frame.is_frameset())
    except Exception:
        return False


def is_motion_frame(frame: Any) -> bool:
    try:
        return bool(frame.is_motion_frame())
    except Exception:
        return False


def as_frameset(frame: Any) -> Any | None:
    try:
        return frame.as_frameset()
    except Exception:
        return None


def frameset_has_color_and_depth(rs: Any, frame: Any) -> bool:
    try:
        return bool(frame.first_or_default(rs.stream.color)) and bool(
            frame.first_or_default(rs.stream.depth)
        )
    except Exception:
        try:
            return bool(frame.get_color_frame()) and bool(frame.get_depth_frame())
        except Exception:
            return False


def frameset_has_color(rs: Any, frame: Any) -> bool:
    try:
        return bool(frame.first_or_default(rs.stream.color))
    except Exception:
        try:
            return bool(frame.get_color_frame())
        except Exception:
            return False


def nearest_index_by_timestamp(sorted_timestamps: list[float], target_ts: float) -> int:
    """Index into sorted_timestamps whose value is closest to target_ts.
    sorted_timestamps must already be sorted ascending. Shared by every
    "match sample stream X to recorded frames" step (IMU here; servo
    position in output_script/synced_capture.py) -- they all need the same
    nearest-of-the-two-bisect-candidates lookup, just against different
    timestamp domains and row shapes."""
    idx = bisect.bisect_left(sorted_timestamps, target_ts)
    candidates = [i for i in (idx - 1, idx) if 0 <= i < len(sorted_timestamps)]
    return min(candidates, key=lambda i: abs(sorted_timestamps[i] - target_ts))


def match_samples_to_frames(
    frame_rows: list[dict[str, Any]], samples: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not samples:
        return []

    sample_timestamps = [s["timestamp_seconds"] for s in samples]
    matched: list[dict[str, Any]] = []
    for frame in frame_rows:
        frame_ts = frame["color_timestamp_seconds"]
        best_idx = nearest_index_by_timestamp(sample_timestamps, frame_ts)
        sample = samples[best_idx]
        matched.append(
            {
                "frame_id": frame["frame_id"],
                "frame_timestamp_seconds": frame_ts,
                "frame_number": sample["frame_number"],
                "timestamp_seconds": sample["timestamp_seconds"],
                "rs_timestamp_ms": sample["rs_timestamp_ms"],
                "timestamp_domain": sample["timestamp_domain"],
                "host_time_ns": sample["host_time_ns"],
                "x": sample["x"],
                "y": sample["y"],
                "z": sample["z"],
            }
        )
    return matched


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
