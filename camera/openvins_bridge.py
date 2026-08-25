"""Thin ctypes wrapper around OpenVINS's ov_capi C ABI (see
open_vins/ov_msckf/src/ov_capi.h/.cpp), for embedding a live OpenVINS tracking
session inside this process -- fed frame-by-frame/sample-by-sample from
whatever already owns the camera (RealSenseCapture), rather than OpenVINS
opening its own camera connection.

IMPORTANT: run this under a venv/interpreter that is NOT conda's Python (or
any Python with $ORIGIN/../lib baked into its own RPATH) -- see this
project's .venv. Conda's python3 binary carries an old-style DT_RPATH
pointing at its own lib/ dir, which (per ELF loading rules) is inherited by
every library it loads afterward, including this one via ctypes -- so it
ends up preferring conda's own (older, ABI-mismatched) libtiff/libgdal over
the system ones this .so was actually built against, and dlopen fails with
an undefined-symbol error. A plain venv created from system python3 has no
such rpath and loads cleanly.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional

# Adjust if open_vins is built somewhere else, or override by passing
# lib_path explicitly to OpenVinsTracker().
DEFAULT_LIB_PATH = Path("/home/hakan/Desktop/umi/open_vins/ov_msckf/build/libov_capi.so")


class OpenVinsTracker:
    """One instance = one independent OpenVINS tracking session (its own
    filter state, starting fresh from construction). To start over (e.g.
    between recording episodes), close() this one and construct a new
    instance -- VioManager itself has no in-place reset."""

    def __init__(self, config_path: str | Path, lib_path: str | Path = DEFAULT_LIB_PATH):
        self._lib = ctypes.CDLL(str(lib_path))
        self._lib.ov_create.argtypes = [ctypes.c_char_p]
        self._lib.ov_create.restype = ctypes.c_void_p
        self._lib.ov_feed_imu.argtypes = [
            ctypes.c_void_p, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ]
        self._lib.ov_feed_imu.restype = None
        self._lib.ov_feed_camera.argtypes = [
            ctypes.c_void_p, ctypes.c_double, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte),
        ]
        self._lib.ov_feed_camera.restype = None
        self._lib.ov_get_pose.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ]
        self._lib.ov_get_pose.restype = ctypes.c_int
        self._lib.ov_destroy.argtypes = [ctypes.c_void_p]
        self._lib.ov_destroy.restype = None

        self._handle = self._lib.ov_create(str(config_path).encode("utf-8"))
        if not self._handle:
            raise RuntimeError(f"ov_create failed to parse config: {config_path}")

        # Fusion state for feed_imu_gyro/feed_imu_accel below -- the D435I
        # delivers gyro and accel as two independent streams, not a single
        # synced IMU topic (same situation as run_realsense_vio.cpp). We pair
        # each accel sample with the most recent gyro sample (nearest-
        # neighbor fusion, not interpolation) and emit one OpenVINS IMU
        # measurement per accel sample.
        self._have_gyro = False
        self._last_gyro = (0.0, 0.0, 0.0)

        self._xyz_buf = (ctypes.c_double * 3)()
        self._quat_buf = (ctypes.c_double * 4)()

    def feed_imu_gyro(self, wx: float, wy: float, wz: float) -> None:
        """Record the latest gyro reading. Call this whenever a gyro sample
        arrives; does not by itself feed OpenVINS (see feed_imu_accel)."""
        self._last_gyro = (wx, wy, wz)
        self._have_gyro = True

    def feed_imu_accel(self, timestamp: float, ax: float, ay: float, az: float) -> None:
        """Call whenever an accel sample arrives. Pairs it with the latest
        gyro reading (feed_imu_gyro) and feeds OpenVINS one IMU measurement.
        No-op until at least one gyro reading has been seen."""
        if not self._have_gyro:
            return
        wx, wy, wz = self._last_gyro
        self._lib.ov_feed_imu(self._handle, timestamp, wx, wy, wz, ax, ay, az)

    def feed_camera_gray(self, timestamp: float, gray) -> None:
        """gray: a 2D (H,W) uint8 numpy array, contiguous, single-channel."""
        h, w = gray.shape[:2]
        ptr = gray.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        self._lib.ov_feed_camera(self._handle, timestamp, w, h, ptr)

    def get_pose(self) -> Optional[tuple]:
        """Returns (x, y, z, qx, qy, qz, qw) if the filter is initialized,
        else None. Call after feed_camera_gray to get the pose as of that
        frame."""
        ok = self._lib.ov_get_pose(self._handle, self._xyz_buf, self._quat_buf)
        if not ok:
            return None
        return tuple(self._xyz_buf) + tuple(self._quat_buf)

    def close(self) -> None:
        if self._handle:
            self._lib.ov_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "OpenVinsTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
