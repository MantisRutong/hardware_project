"""Thin ctypes wrapper around ORB-SLAM3's orb_capi C ABI (see
ORB_SLAM3/include/orb_capi.h / src/orb_capi.cc), for embedding a live
ORB-SLAM3 Stereo-Inertial tracking session inside this process -- fed
frame-by-frame/sample-by-sample from whatever already owns the camera
(RealSenseCapture), rather than ORB-SLAM3 opening its own camera connection.
Mirrors openvins_bridge.py's OpenVinsTracker, same idea, different backend.

Why ORB-SLAM3 instead of (or alongside) OpenVINS for live feedback: extensive
empirical testing (see open_vins/ this project's history) showed monocular
OpenVINS drifts unboundedly under continuous handheld motion with no pauses
(ZUPT needs a stationary moment to correct, and none exist during continuous
motion) -- up to thousands of meters in a dedicated stress test. ORB-SLAM3's
Stereo-Inertial mode gets real metric scale directly from the known IR
stereo baseline instead of estimating it online, which eliminated that
failure mode in equivalent live testing (bounded, sane trajectories over
50+ seconds of continuous motion with zero pauses). See synced_capture.py's
--orbslam flag, which is the default live tracker as of this integration.

IMPORTANT: run this under a venv/interpreter that is NOT conda's Python (or
any Python with $ORIGIN/../lib baked into its own RPATH) -- see this
project's .venv and openvins_bridge.py's docstring for why (same root cause,
same fix, applies to any compiled .so loaded via ctypes on this machine).
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional

# Adjust if ORB_SLAM3 is built somewhere else, or override by passing
# lib_path explicitly to OrbSlamTracker().
DEFAULT_LIB_PATH = Path("/home/hakan/Desktop/umi/ORB_SLAM3/lib/liborb_capi.so")
DEFAULT_VOCAB_PATH = Path("/home/hakan/Desktop/umi/ORB_SLAM3/Vocabulary/ORBvoc.txt")
DEFAULT_SETTINGS_PATH = Path("/home/hakan/Desktop/umi/ORB_SLAM3/config/RealSense_D435i_ours.yaml")


class OrbSlamTracker:
    """One instance = one independent ORB-SLAM3 Stereo-Inertial tracking
    session (its own map/pose graph, starting fresh from construction). To
    start over (e.g. between recording episodes), close() this one and
    construct a new instance -- ORB_SLAM3::System has a Reset()/
    ResetActiveMap() API but this wrapper doesn't expose it, matching how
    OpenVinsTracker handles the same situation (VioManager has no in-place
    reset either) -- a fresh handle per episode keeps both trackers'
    lifecycle identical from synced_capture.py's point of view.

    Unlike OpenVinsTracker (which buffers accel/gyro separately and fuses on
    every accel sample), ORB-SLAM3 consumes a batch of IMU points per
    tracked frame -- see orb_capi.h. feed_imu() just appends to that internal
    C++-side buffer; track_stereo() drains it and runs tracking in one call.
    """

    def __init__(
        self,
        settings_path: str | Path = DEFAULT_SETTINGS_PATH,
        vocab_path: str | Path = DEFAULT_VOCAB_PATH,
        lib_path: str | Path = DEFAULT_LIB_PATH,
        use_viewer: bool = False,
    ):
        self._lib = ctypes.CDLL(str(lib_path))
        self._lib.orb_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        self._lib.orb_create.restype = ctypes.c_void_p
        self._lib.orb_feed_imu.argtypes = [
            ctypes.c_void_p, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ]
        self._lib.orb_feed_imu.restype = None
        self._lib.orb_track_stereo.argtypes = [
            ctypes.c_void_p, ctypes.c_double, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_long),
        ]
        self._lib.orb_track_stereo.restype = ctypes.c_int
        self._lib.orb_shutdown.argtypes = [ctypes.c_void_p]
        self._lib.orb_shutdown.restype = None
        for _name, _restype in (("orb_get_current_map_kf_count", ctypes.c_long),
                                 ("orb_current_map_imu_initialized", ctypes.c_int),
                                 ("orb_current_map_imu_ba2", ctypes.c_int)):
            getattr(self._lib, _name).argtypes = [ctypes.c_void_p]
            getattr(self._lib, _name).restype = _restype
        self._lib.orb_get_current_map_id.argtypes = [ctypes.c_void_p]
        self._lib.orb_get_current_map_id.restype = ctypes.c_long
        self._lib.orb_save_trajectory_tum.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.orb_save_trajectory_tum.restype = None
        self._lib.orb_destroy.argtypes = [ctypes.c_void_p]
        self._lib.orb_destroy.restype = None

        self._handle = self._lib.orb_create(
            str(vocab_path).encode("utf-8"), str(settings_path).encode("utf-8"), 1 if use_viewer else 0
        )
        if not self._handle:
            raise RuntimeError(
                f"orb_create failed (vocab={vocab_path}, settings={settings_path}) -- "
                "check both paths exist and the settings YAML is valid."
            )
        self._shut_down = False

        self._xyz_buf = (ctypes.c_double * 3)()
        self._quat_buf = (ctypes.c_double * 4)()
        self._map_epoch_buf = ctypes.c_long(0)

    def feed_imu(self, timestamp: float, gx: float, gy: float, gz: float, ax: float, ay: float, az: float) -> None:
        """Buffer one IMU sample (gyro rad/s, accel m/s^2). Consumed by the
        next track_stereo() call. Unlike OpenVinsTracker's feed_imu_gyro/
        feed_imu_accel split, this expects one already-paired sample per
        call -- see RealSenseCapture's on_imu_sample hook, which fires
        separately for gyro/accel; the caller (synced_capture.py) does the
        same nearest-gyro pairing before calling this, mirroring how
        run_realsense_vio.cpp/the stock ORB-SLAM3 D435i example both handle
        the D435i's two independent gyro/accel streams."""
        self._lib.orb_feed_imu(self._handle, timestamp, gx, gy, gz, ax, ay, az)

    def track_stereo(self, timestamp: float, left_gray, right_gray) -> Optional[tuple]:
        """left_gray/right_gray: 2D (H,W) uint8 numpy arrays, contiguous,
        single-channel, same shape, already rectified to match the settings
        file's calibration. Returns (x, y, z, qx, qy, qz, qw) -- the left
        camera's pose in the world frame -- if tracking succeeded this
        frame, else None (lost / not yet initialized). Use last_map_epoch
        (updated on every call, tracked or not) to detect resets -- see its
        docstring."""
        h, w = left_gray.shape[:2]
        left_ptr = left_gray.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        right_ptr = right_gray.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        ok = self._lib.orb_track_stereo(
            self._handle, timestamp, w, h, left_ptr, right_ptr,
            self._xyz_buf, self._quat_buf, ctypes.byref(self._map_epoch_buf)
        )
        if not ok:
            return None
        return tuple(self._xyz_buf) + tuple(self._quat_buf)

    @property
    def last_map_epoch(self) -> int:
        """System::GetMapEpoch() as of the most recent track_stereo() call
        (tracked or not -- always available). A change in this value between
        two calls means ORB-SLAM3 just reset in a way that can move the
        active map's pose origin -- e.g. after "Not enough motion for
        initializing. Reseting..." or "IMU is not or recently initialized.
        Reseting active map...". NOT simply "the active map's ID changed":
        live testing showed the common reset path clears and re-initializes
        the SAME map object in place (its own ID never changes) -- see
        System.h's GetMapEpoch() doc comment for the full story. Do not
        treat poses across an epoch change as a continuous trajectory. 0
        before the first call."""
        return self._map_epoch_buf.value

    @property
    def current_map_id(self) -> int:
        """The id of the map tracking is currently in.

        Answers "am I in the loaded atlas yet", which no other signal here
        does. LoadAtlas restores the saved map as id 0, but the session then
        cold-starts its OWN map (id 1) and tracks in that -- the very cold
        start the atlas exists to avoid -- until LoopClosing recognises the
        region and merges, at which point Atlas::ChangeMap switches the
        active map and this becomes 0 again.

        So neither of the obvious signals answers the question:
        track_stereo() returning a pose is equally true of the cold-started
        map, and last_map_epoch never moves for a merge (a merge is not a
        reset -- see that property's docstring). Watch this instead.

        Unlike last_map_epoch, this queries ORB-SLAM3 on every access rather
        than reading a value cached by the last track_stereo() call, since a
        merge happens on the LoopClosing thread and can land between frames.
        0 with no handle, or after shutdown()."""
        if not self._handle:
            return 0
        return int(self._lib.orb_get_current_map_id(self._handle))

    @property
    def place_recognition_gates(self) -> dict:
        """The three conditions LoopClosing::NewDetectCommonRegions returns
        early on, before it will look for a place it recognises.

        Place recognition is the only route into a loaded atlas, so when
        current_map_id never changes these say which gate is holding it
        shut:

            imu_ba2 False        -> gate 1, the map has not finished IMU BA2
            keyframes < 5 or 12  -> gates 2 and 3, the map is too small

        All three are about the CURRENT map -- the one this session
        cold-started -- never the loaded atlas. Diagnostic only; nothing in
        the pipeline branches on it."""
        if not self._handle:
            return {"keyframes": 0, "imu_initialized": False, "imu_ba2": False}
        return {
            "keyframes": int(self._lib.orb_get_current_map_kf_count(self._handle)),
            "imu_initialized": bool(self._lib.orb_current_map_imu_initialized(self._handle)),
            "imu_ba2": bool(self._lib.orb_current_map_imu_ba2(self._handle)),
        }

    def shutdown(self) -> None:
        """Stops ORB-SLAM3's internal threads. Call before save_trajectory_tum.
        Safe to call more than once; close() calls this automatically if not
        already done."""
        if self._handle and not self._shut_down:
            self._lib.orb_shutdown(self._handle)
            self._shut_down = True

    def save_trajectory_tum(self, path: str | Path) -> None:
        """Call after shutdown(). Writes TUM format: timestamp x y z qx qy qz qw."""
        if self._handle:
            self._lib.orb_save_trajectory_tum(self._handle, str(path).encode("utf-8"))

    def close(self) -> None:
        if self._handle:
            self._lib.orb_destroy(self._handle)  # shuts down internally if needed
            self._handle = None

    def __enter__(self) -> "OrbSlamTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
