"""Live gripper-width readout, for checking (and fine-tuning) the
calibration in output_script/synced_capture.py against a physical ruler.

Torque stays OFF the whole time (like spring_position_mode.py's
--measure-freeplay) -- safe to move the gripper by hand while this runs.
Prints raw servo angle alongside the calibrated width (mm) it maps to, so
you can hold the gripper open to a few known widths (a ruler/calipers
against the jaws) and see whether the printed number matches.

Deliberately self-contained rather than importing output_script/
synced_capture.py: that module unconditionally imports orbslam_bridge/
openvins_bridge at the top, which ctypes.CDLL-load compiled .so libraries
immediately on import -- total overkill (and a source of fragile failures,
e.g. under conda's Python) for a simple gripper-width check that has
nothing to do with the camera or SLAM. The calibration loading/interpolation
logic below is a small, deliberate duplicate of load_gripper_calibration/
interp_gripper_width_mm there -- if you change the interpolation behavior
there, mirror it here too.

Usage:
    python3 check_gripper_width.py
    python3 check_gripper_width.py --offset-deg -3.5   # try a different bias while comparing to the ruler
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from spring_position_mode import DynamixelPositionSpring, resolve_port

# Mirrors output_script/synced_capture.py's DEFAULT_GRIPPER_CALIBRATION_CSV /
# GRIPPER_CALIBRATION_BIAS_DEG / GRIPPER_MAX_WIDTH_MM -- see that file for
# the canonical, actively-used values and their own provenance comments.
# Keep these in sync by hand if either changes.
DEFAULT_CALIBRATION_CSV = Path(__file__).resolve().parent.parent / "mapping_csv" / "mapping_function.csv"
DEFAULT_OFFSET_DEG = 0.24  # re-measured and ruler-verified 2026-08-26 after gripper reassembly -- see synced_capture.py
DEFAULT_MAX_WIDTH_MM = 80.0


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    pairs = sorted((float(r["gear_displacement"]), float(r["gripper_displacement"])) for r in rows)
    return np.array([p for p, _ in pairs]), np.array([w for _, w in pairs])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=None, help="Serial port. Auto-detected if omitted.")
    parser.add_argument("--baud", type=int, default=57600)
    parser.add_argument("--id", type=int, default=1, dest="dxl_id")
    parser.add_argument("--calibration-csv", type=Path, default=DEFAULT_CALIBRATION_CSV)
    parser.add_argument("--offset-deg", type=float, default=DEFAULT_OFFSET_DEG,
                         help=f"Added to every gear_displacement value before interpolating -- see "
                              f"GRIPPER_CALIBRATION_BIAS_DEG in output_script/synced_capture.py. "
                              f"Default {DEFAULT_OFFSET_DEG:.2f}.")
    parser.add_argument("--max-width-mm", type=float, default=DEFAULT_MAX_WIDTH_MM)
    parser.add_argument("--duration", type=float, default=None, help="Run time in seconds (default: run until Ctrl+C)")
    args = parser.parse_args()

    positions, widths = load_calibration(args.calibration_csv)
    positions = positions + args.offset_deg

    port_name = resolve_port(args.port)
    servo = DynamixelPositionSpring(port_name, args.baud, args.dxl_id)
    servo.set_torque(False)
    print(f"Loaded {len(positions)} calibration points from {args.calibration_csv}, offset {args.offset_deg:.2f} deg.")
    print("Torque OFF -- move the gripper freely by hand. Hold at a few known openings and compare to a ruler.")
    print("Ctrl+C to stop.\n")

    start_time = time.monotonic()
    try:
        while args.duration is None or (time.monotonic() - start_time) < args.duration:
            angle = servo.read_position_deg()
            width = min(float(np.interp(angle, positions, widths)), args.max_width_mm)
            print(f"\rraw_angle={angle:8.2f} deg   gripper_width={width:6.2f} mm   ", end="", flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        servo.close()
        print("\n\nTorque disabled, port closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
