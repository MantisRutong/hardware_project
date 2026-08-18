"""Make a physical XL330-M077 behave like a torsional spring using the
servo's own built-in position control loop, instead of computing current
externally over USB every cycle.

Background (see spring_control.py's docstring for the full story): an
external loop that reads position over USB and writes current back is
fundamentally rate-limited by the USB link -- on this U2D2/FTDI setup that's
~31Hz with ~32ms of round-trip delay. Multiple externally-computed PD gain
combinations (stiffness 0.4-3.0 mA/deg, damping 0.05-0.8 mA/(deg/s), with
and without output filtering/slew-limiting) were tried and physically
tested in this session; every one of them, physically tested, developed a
self-sustained oscillation after a nudge. The delay itself, not the gain
choice, is the problem.

DYNAMIXEL Current-based Position Control Mode sidesteps this entirely: the
servo's own onboard microcontroller runs a factory-tuned position PID loop
at a high internal rate (no USB round trip in that loop at all). We just:
  1. Set Operating Mode = Current-based Position Control.
  2. Set Goal Current = a torque cap (how hard it can push back, i.e. how
     "strong" the spring feels once displaced a lot).
  3. Set Position P/D gain = how stiff/damped it feels near center.
  4. Set Goal Position = center, ONCE.
Then the servo holds/restores that position on its own. Our script only
polls Present Position occasionally to print status -- that polling rate no
longer has anything to do with stability.

Run scan_servo.py first to confirm port/baud/ID (found on the original
macOS setup this was built on: /dev/cu.usbserial-FTBEQKWH, 57600 baud, ID
1). If --port is omitted, this script auto-detects it the same way
scan_servo.py's port search does, so it should work unmodified on other
machines (e.g. a Raspberry Pi, where the U2D2 usually shows up as
/dev/ttyUSB0 instead) as long as baud/ID are unchanged on the servo itself.

Safety notes:
  - Keep fingers clear of pinch points while torque is enabled.
  - Start with a low --max-current and increase gradually.
  - Ctrl+C always disables torque before exiting.

Usage:
    python3 spring_position_mode.py
    python3 spring_position_mode.py --p-gain 400 --d-gain 500 --max-current 150
    python3 spring_position_mode.py --plot --plot-window 15
"""

from __future__ import annotations

import argparse
import struct
import sys
import time

from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

from scan_servo import find_candidate_ports

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

PROTOCOL_VERSION = 2.0

ADDR_TEMPERATURE_LIMIT = 31    # EEPROM: the actual configured overheat shutoff threshold, deg C
ADDR_OPERATING_MODE = 11
ADDR_HARDWARE_ERROR_STATUS = 70  # bit flags; bit 2 (0x04) = Overheating Error
ADDR_POSITION_D_GAIN = 80
ADDR_POSITION_I_GAIN = 82
ADDR_POSITION_P_GAIN = 84
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_CURRENT = 102
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_TEMPERATURE = 146   # deg C, 1 deg C per LSB

HARDWARE_ERROR_OVERHEATING_BIT = 0x04
TEMPERATURE_WARNING_MARGIN_C = 10.0  # default: start warning this many deg C below the shutoff

# Present Current (2B) + Present Velocity (4B) + Present Position (4B) are
# back-to-back registers (126..135), so all three can be read in a single
# serial transaction. There's no stability reason to do this here (unlike
# spring_control.py, this script isn't in the control loop), it's just
# fewer round trips for the same status read.
ADDR_STATE_BLOCK = ADDR_PRESENT_CURRENT
STATE_BLOCK_LEN = 10

OPERATING_MODE_CURRENT_BASED_POSITION_CONTROL = 5

# Measured mechanical range of motion of THIS gripper (horn angle at each end
# of travel), from a hold-fully-open / hold-fully-closed test with
# --measure-freeplay on 2026-08-17. These are absolute horn angles tied to
# this specific horn-to-gear alignment -- if the horn is ever removed and
# reinstalled (even one spline tooth off), or the gripper's own gearing is
# changed, re-measure with --measure-freeplay and update these two numbers.
GRIPPER_OPEN_DEG = 320.5
GRIPPER_CLOSED_DEG = 216.9
GRIPPER_RANGE_MARGIN_DEG = 0.5  # default safety margin kept away from each end

POSITION_UNIT_DEG = 0.088          # deg per position tick (4096 ticks/rev)
VELOCITY_UNIT_DEG_S = 0.229 * 6.0  # 0.229 rev/min per tick -> deg/s per tick
CURRENT_UNIT_MA = 1.0              # mA per Present/Goal Current tick (XL330 series)

# XL330-M077 has no torque sensor -- Present Current (the actual measured
# armature current) is the real, directly-measured quantity, and torque is
# only proportional to it via the motor's torque constant. This value is a
# NOMINAL estimate from the datasheet's rated torque/rated current ratio,
# not something calibrated against this specific unit -- treat the printed
# torque as approximate, and prefer the printed current (mA) if you need the
# actual measured quantity. Override with --torque-constant if you have a
# better-calibrated value.
NOMINAL_TORQUE_CONSTANT_NM_PER_MA = 0.4 / 1500  # ~ rated stall torque / stall current

# XL330 also has no accelerometer -- acceleration below is derived by
# differentiating two consecutive Present Velocity readings, not measured
# directly. At the ~10Hz status-print rate used here that's a coarse
# derivative; treat it as an estimate, not a precise measurement.


def to_signed(value: int, bits: int) -> int:
    mask = 1 << (bits - 1)
    return (value ^ mask) - mask


def to_unsigned(value: int, bits: int) -> int:
    return value & ((1 << bits) - 1)


class DynamixelPositionSpring:
    def __init__(self, port_name: str, baud: int, dxl_id: int):
        self.dxl_id = dxl_id
        self.port_handler = PortHandler(port_name)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)

        if not self.port_handler.openPort():
            raise RuntimeError(f"Could not open port {port_name}")
        if not self.port_handler.setBaudRate(baud):
            raise RuntimeError(f"Could not set baud rate {baud} on {port_name}")

        model_number, comm_result, error = self.packet_handler.ping(self.port_handler, dxl_id)
        if comm_result != COMM_SUCCESS:
            raise RuntimeError(
                f"Could not ping servo ID {dxl_id} on {port_name} @ {baud}. "
                f"Run scan_servo.py to re-check the port/baud/ID."
            )
        print(f"Connected: ID {dxl_id}, model number {model_number}")

    def _check(self, comm_result: int, error: int, what: str) -> None:
        if comm_result != COMM_SUCCESS:
            raise RuntimeError(f"{what} failed: {self.packet_handler.getTxRxResult(comm_result)}")
        if error != 0:
            raise RuntimeError(f"{what} reported hardware error: {self.packet_handler.getRxPacketError(error)}")

    def _write1(self, addr: int, value: int) -> None:
        result, error = self.packet_handler.write1ByteTxRx(self.port_handler, self.dxl_id, addr, value)
        self._check(result, error, f"write1 addr={addr}")

    def _write2(self, addr: int, value: int) -> None:
        result, error = self.packet_handler.write2ByteTxRx(self.port_handler, self.dxl_id, addr, to_unsigned(value, 16))
        self._check(result, error, f"write2 addr={addr}")

    def _write4(self, addr: int, value: int) -> None:
        result, error = self.packet_handler.write4ByteTxRx(self.port_handler, self.dxl_id, addr, to_unsigned(value, 32))
        self._check(result, error, f"write4 addr={addr}")

    def _read1(self, addr: int) -> int:
        raw, result, error = self.packet_handler.read1ByteTxRx(self.port_handler, self.dxl_id, addr)
        self._check(result, error, f"read1 addr={addr}")
        return raw

    def _read2(self, addr: int) -> int:
        raw, result, error = self.packet_handler.read2ByteTxRx(self.port_handler, self.dxl_id, addr)
        self._check(result, error, f"read2 addr={addr}")
        return raw

    def set_torque(self, enable: bool) -> None:
        self._write1(ADDR_TORQUE_ENABLE, 1 if enable else 0)

    def read_temperature_limit_c(self) -> int:
        """The actual configured overheat shutoff threshold (not a guess)."""
        return self._read1(ADDR_TEMPERATURE_LIMIT)

    def read_temperature_c(self) -> int:
        return self._read1(ADDR_PRESENT_TEMPERATURE)

    def read_hardware_error(self) -> int:
        return self._read1(ADDR_HARDWARE_ERROR_STATUS)

    def read_position_deg(self) -> float:
        raw, result, error = self.packet_handler.read4ByteTxRx(self.port_handler, self.dxl_id, ADDR_PRESENT_POSITION)
        self._check(result, error, "read present position")
        return to_signed(raw, 32) * POSITION_UNIT_DEG

    def read_state(self) -> tuple[float, float, float]:
        """One serial transaction: returns (current_ma, velocity_deg_s, position_deg)."""
        data, result, error = self.packet_handler.readTxRx(self.port_handler, self.dxl_id, ADDR_STATE_BLOCK, STATE_BLOCK_LEN)
        self._check(result, error, "read state block")
        current_raw, velocity_raw, position_raw = struct.unpack("<hii", bytes(data))
        return (
            current_raw * CURRENT_UNIT_MA,
            velocity_raw * VELOCITY_UNIT_DEG_S,
            position_raw * POSITION_UNIT_DEG,
        )

    def read_gains(self) -> tuple[int, int, int]:
        return self._read2(ADDR_POSITION_P_GAIN), self._read2(ADDR_POSITION_I_GAIN), self._read2(ADDR_POSITION_D_GAIN)

    def write_goal_position_deg(self, angle_deg: float) -> None:
        self._write4(ADDR_GOAL_POSITION, int(round(angle_deg / POSITION_UNIT_DEG)))

    def configure_spring(self, p_gain: int, i_gain: int, d_gain: int, max_current_ma: float, center_deg: float) -> None:
        self.set_torque(False)
        self._write1(ADDR_OPERATING_MODE, OPERATING_MODE_CURRENT_BASED_POSITION_CONTROL)
        self._write2(ADDR_POSITION_P_GAIN, p_gain)
        self._write2(ADDR_POSITION_I_GAIN, i_gain)
        self._write2(ADDR_POSITION_D_GAIN, d_gain)
        self._write2(ADDR_GOAL_CURRENT, int(round(max_current_ma / CURRENT_UNIT_MA)))
        self.set_torque(True)
        self.write_goal_position_deg(center_deg)

    def close(self) -> None:
        try:
            self.set_torque(False)
        except Exception:
            pass
        self.port_handler.closePort()


class LivePlot:
    """Rolling real-time plot of position error / velocity / acceleration /
    current, updated in lockstep with the polling loop while the servo is
    running (not a replay after the fact)."""

    def __init__(self, window_seconds: float = 10.0):
        self.window_seconds = window_seconds
        self.times: list[float] = []
        self.errors: list[float] = []
        self.velocities: list[float] = []
        self.accelerations: list[float] = []
        self.currents: list[float] = []
        self.closed = False

        plt.ion()
        self.fig, self.axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
        titles = [
            "Position error (deg)",
            "Velocity (deg/s)",
            "Acceleration (deg/s^2)",
            "Current (mA, proportional to torque)",
        ]
        self.lines = []
        for ax, title in zip(self.axes, titles):
            (line,) = ax.plot([], [])
            ax.set_ylabel(title)
            ax.grid(True)
            self.lines.append(line)
        self.axes[-1].set_xlabel("Time (s)")
        self.fig.suptitle("XL330 spring -- live")
        self.fig.tight_layout()
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self.fig.canvas.draw()
        plt.pause(0.01)

    def _on_close(self, _event) -> None:
        self.closed = True

    def update(self, t: float, error: float, velocity: float, accel: float, current_ma: float) -> None:
        self.times.append(t)
        self.errors.append(error)
        self.velocities.append(velocity)
        self.accelerations.append(accel)
        self.currents.append(current_ma)

        cutoff = t - self.window_seconds
        while self.times and self.times[0] < cutoff:
            self.times.pop(0)
            self.errors.pop(0)
            self.velocities.pop(0)
            self.accelerations.pop(0)
            self.currents.pop(0)

        series = [self.errors, self.velocities, self.accelerations, self.currents]
        for line, ax, ys in zip(self.lines, self.axes, series):
            line.set_data(self.times, ys)
            ax.relim()
            ax.autoscale_view()

        if len(self.times) >= 2:
            for ax in self.axes:
                ax.set_xlim(self.times[0], self.times[-1])

        self.fig.canvas.draw_idle()

    def pause(self, seconds: float) -> None:
        # plt.pause() both sleeps AND lets the GUI event loop redraw/respond
        # to input (e.g. the window close button) -- a plain time.sleep()
        # here would freeze the plot window for the duration.
        plt.pause(seconds)


def duration_active(start_time: float, duration: float | None) -> bool:
    """True while a --duration-limited loop should keep running (forever if duration is None)."""
    return duration is None or (time.monotonic() - start_time) < duration


def run_measure_freeplay(servo: DynamixelPositionSpring, args: argparse.Namespace) -> int:
    """Diagnostic mode: torque stays OFF, we just poll Present Position and track
    the running min/max while you move the mechanism by hand -- reads the actual
    encoder instead of eyeballing tiny movements.

    Recommended technique: hold the downstream side (e.g. the gripper rack)
    perfectly still with one hand, then gently rock the horn back and forth by
    hand with the other. The horn can move within the backlash gap without
    dragging the rack at all -- the min/max range reported here IS that
    backlash, in horn-degrees.

    (If you instead drag the whole mechanism through its full range of motion,
    this reports total travel range, not backlash -- a different, also useful,
    number, just don't confuse the two.)
    """
    print("Freeplay measurement mode -- torque stays OFF, nothing is being commanded.")
    print("Hold the gripper rack perfectly still, then gently rock the horn by hand;")
    print("the min/max range below is the backlash, in horn-degrees.")
    print("Ctrl+C (or wait for --duration) to stop and see the summary.\n")

    start_time = time.monotonic()
    min_angle = max_angle = None
    try:
        while duration_active(start_time, args.duration):
            angle = servo.read_position_deg()
            min_angle = angle if min_angle is None else min(min_angle, angle)
            max_angle = angle if max_angle is None else max(max_angle, angle)
            print(f"\rangle={angle:8.2f} deg   min={min_angle:8.2f}  max={max_angle:8.2f}  "
                  f"range={max_angle - min_angle:6.2f} deg   ", end="", flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass

    if min_angle is None:
        print("\nNo samples read.")
        return 0

    print(f"\n\nFinal range: {max_angle - min_angle:.2f} deg (min={min_angle:.2f}, max={max_angle:.2f})")
    print("If that was the 'hold rack still, rock the horn' test, that range is the backlash.")
    return 0


def run_spring(servo: DynamixelPositionSpring, args: argparse.Namespace) -> int:
    try:
        factory_p, factory_i, factory_d = servo.read_gains()
        print(f"Factory Position PID gains were: P={factory_p} I={factory_i} D={factory_d}")

        temperature_limit_c = servo.read_temperature_limit_c()
        start_temperature_c = servo.read_temperature_c()
        print(f"Present temperature: {start_temperature_c} deg C "
              f"(configured overheat shutoff: {temperature_limit_c} deg C)")

        start_angle = servo.read_position_deg()
        if args.center_at_open:
            center = args.gripper_open_deg
            print(f"Starting angle: {start_angle:.2f} deg, spring center: {center:.2f} deg "
                  f"(pinned to calibrated fully-open angle, ignoring --center)")
        elif args.center_at_closed:
            center = args.gripper_closed_deg
            print(f"Starting angle: {start_angle:.2f} deg, spring center: {center:.2f} deg "
                  f"(pinned to calibrated fully-closed angle, ignoring --center)")
        elif args.center is not None:
            center = start_angle + args.center
            print(f"Starting angle: {start_angle:.2f} deg, spring center: {center:.2f} deg")
        else:
            # Default when nothing else is specified: pin to the calibrated
            # fully-open angle rather than wherever the horn happens to be
            # sitting when the script starts (that used to be the default,
            # but it made "spring back to fully open" depend on luck).
            center = args.gripper_open_deg
            print(f"Starting angle: {start_angle:.2f} deg, spring center: {center:.2f} deg "
                  f"(default: pinned to calibrated fully-open angle)")

        # Keep the commanded goal position away from the gripper's own physical
        # end stops. Unlike a hand twist (which the mechanism itself stops), a
        # goal position beyond the real range of motion is a standing command --
        # the servo would keep straining against the hard stop indefinitely,
        # even sitting untouched, instead of just failing to reach it once.
        if not args.ignore_range_limit:
            safe_min = min(args.gripper_open_deg, args.gripper_closed_deg) + args.range_margin
            safe_max = max(args.gripper_open_deg, args.gripper_closed_deg) - args.range_margin
            if center < safe_min or center > safe_max:
                clamped = min(max(center, safe_min), safe_max)
                print(f"Warning: spring center {center:.2f} deg is outside the safe range "
                      f"[{safe_min:.2f}, {safe_max:.2f}] deg (measured gripper travel "
                      f"{args.gripper_closed_deg:.1f}-{args.gripper_open_deg:.1f} deg, minus a "
                      f"{args.range_margin:.1f} deg margin) -- clamping to {clamped:.2f} deg "
                      f"so it doesn't hold torque against the mechanical stop forever. "
                      f"Pass --ignore-range-limit to disable this.")
                center = clamped

        print(f"Setting Position P={args.p_gain} I={args.i_gain} D={args.d_gain}, "
              f"max current (torque cap) {args.max_current} mA")

        servo.configure_spring(args.p_gain, args.i_gain, args.d_gain, args.max_current, center)
        print("Torque enabled, goal position set once. The servo's own onboard loop "
              "now holds/restores this position -- twist the horn to feel the spring. "
              "Ctrl+C to stop.")
        print("Note: XL330 has no acceleration or torque sensor. Acceleration is derived "
              "by differentiating velocity between reads; torque is estimated from measured "
              f"current using an approximate constant ({args.torque_constant:.6g} N*m/mA) -- "
              "prefer the printed current (mA) if you need the actual measured quantity.\n")

        live_plot = None
        if args.plot:
            if plt is None:
                print("--plot requested but matplotlib is not installed; continuing with text "
                      "output only. Install with `pip install matplotlib`.")
            else:
                try:
                    live_plot = LivePlot(window_seconds=args.plot_window)
                except Exception as exc:
                    print(f"--plot requested but the plot window could not be created ({exc}); "
                          "continuing with text output only. (This usually means no display is "
                          "available, e.g. an SSH session without X forwarding.)")
                    live_plot = None

        start_time = time.monotonic()
        last_sample_time = start_time
        last_temp_check_time = start_time - 1.0  # force an immediate first check
        prev_velocity = None
        temperature_c = start_temperature_c
        while duration_active(start_time, args.duration):
            now = time.monotonic()
            dt = now - last_sample_time
            last_sample_time = now
            t = now - start_time

            current_ma, velocity, angle = servo.read_state()
            error = angle - center
            torque_nm = current_ma * args.torque_constant
            if prev_velocity is None:
                accel = 0.0
            else:
                accel = (velocity - prev_velocity) / dt
            prev_velocity = velocity

            # Temperature changes slowly -- checking once/sec (instead of
            # every ~100ms loop tick) is plenty and cuts serial traffic.
            if now - last_temp_check_time >= 1.0:
                last_temp_check_time = now
                temperature_c = servo.read_temperature_c()
                near_limit = temperature_c >= temperature_limit_c - args.temp_warning_margin
                # Overheat protection can't have tripped while comfortably below
                # the limit, so skip this second round trip until it matters.
                hardware_error = servo.read_hardware_error() if near_limit else 0
                if hardware_error & HARDWARE_ERROR_OVERHEATING_BIT:
                    print(f"\n!!! OVERHEATING PROTECTION TRIGGERED by the servo itself at "
                          f"{temperature_c} deg C -- torque has been cut. Let it cool down. !!!")
                elif near_limit:
                    print(f"\nWarning: temperature {temperature_c} deg C is within "
                          f"{args.temp_warning_margin:.0f} deg C of the {temperature_limit_c} deg C "
                          f"overheat shutoff -- consider a break.")

            print(f"\rangle={angle:8.2f} deg  error={error:8.2f} deg  "
                  f"vel={velocity:8.2f} deg/s  accel={accel:9.1f} deg/s^2  "
                  f"current={current_ma:6.1f} mA  torque~={torque_nm:6.3f} N*m  "
                  f"temp={temperature_c:3d} C   ",
                  end="", flush=True)

            if live_plot is not None:
                live_plot.update(t, error, velocity, accel, current_ma)
                if live_plot.closed:
                    print("\nPlot window closed, stopping.")
                    break
                live_plot.pause(0.1)
            else:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C).")

    return 0


def run(args: argparse.Namespace) -> int:
    servo = DynamixelPositionSpring(args.port, args.baud, args.id)
    try:
        if args.measure_freeplay:
            return run_measure_freeplay(servo, args)
        return run_spring(servo, args)
    finally:
        servo.close()
        print("\nTorque disabled, port closed.")


def resolve_port(explicit_port: str | None) -> str:
    if explicit_port:
        return explicit_port
    candidates = find_candidate_ports()
    if len(candidates) == 1:
        print(f"Auto-detected port: {candidates[0]}")
        return candidates[0]
    if not candidates:
        raise RuntimeError(
            "No --port given and no USB-serial port auto-detected. "
            "Run scan_servo.py to find/confirm the port, then pass it with --port."
        )
    raise RuntimeError(
        f"No --port given and multiple candidate ports found: {candidates}. "
        f"Run scan_servo.py to identify the right one, then pass it with --port."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=None,
                         help="Serial port for the U2D2. Auto-detected if omitted (macOS: /dev/cu.usbserial-*; "
                              "Linux/Raspberry Pi: /dev/ttyUSB*), as long as exactly one candidate is found.")
    parser.add_argument("--baud", type=int, default=57600, help="Baud rate")
    parser.add_argument("--id", type=int, default=1, help="DYNAMIXEL ID")
    parser.add_argument("--p-gain", type=int, default=100, dest="p_gain",
                         help="Internal Position P Gain (raw register units) -- the 'spring constant'. "
                              "Higher = stiffer/snappier near center, but more effort to twist by hand. "
                              "(Started at 400, halved to 200, then halved again to 100 after feeling too "
                              "stiff even at slow twist speeds.)")
    parser.add_argument("--i-gain", type=int, default=30, dest="i_gain",
                         help="Internal Position I Gain. A real unforced spring settles with zero error, but "
                              "P+D alone can get stuck a few degrees short of the goal against static "
                              "friction -- a small I gain pushes through that over a second or so. Set to 0 "
                              "to disable (pure P+D), which trades that reliability for zero holding current "
                              "once released, but can visibly stall short of the goal.")
    parser.add_argument("--d-gain", type=int, default=1200, dest="d_gain",
                         help="Internal Position D Gain (raw register units) -- the damping factor. "
                              "Higher = more damped/less overshoot. (Raised from the original 800 default.)")
    parser.add_argument("--max-current", type=float, default=150.0, dest="max_current",
                         help="Torque cap, mA -- how hard it can push back at large displacement (safety limit too).")
    parser.add_argument("--center", type=float, default=None,
                         help="Spring center offset in degrees from the starting position. Passing this "
                              "switches to relative-to-start mode instead of the default (pinned to the "
                              "calibrated fully-open angle, see --center-at-open).")
    parser.add_argument("--center-at-open", action="store_true", dest="center_at_open",
                         help="Pin the spring center to the calibrated fully-open angle (--gripper-open-deg) "
                              "instead of an offset from wherever the horn happens to be when the script "
                              "starts. This is already the default with no centering flags at all -- pass it "
                              "explicitly only for clarity. Mutually exclusive with --center-at-closed. The "
                              "--range-margin clamp still applies on top of this.")
    parser.add_argument("--center-at-closed", action="store_true", dest="center_at_closed",
                         help="Same as --center-at-open but pins to the calibrated fully-closed angle "
                              "(--gripper-closed-deg).")
    parser.add_argument("--duration", type=float, default=None, help="Run time in seconds (default: run until Ctrl+C)")
    parser.add_argument("--torque-constant", type=float, default=NOMINAL_TORQUE_CONSTANT_NM_PER_MA,
                         dest="torque_constant",
                         help="N*m of torque per mA of current, used only to print an estimated torque. "
                              "Default is a nominal datasheet estimate, not calibrated to this unit -- "
                              "the printed current (mA) is the actual measured quantity.")
    parser.add_argument("--gripper-open-deg", type=float, default=GRIPPER_OPEN_DEG, dest="gripper_open_deg",
                         help=f"Horn angle (deg) at the gripper's fully-open end stop, from a "
                              f"--measure-freeplay hold-open/hold-closed calibration. Default is the value "
                              f"measured for this assembly on 2026-08-17 ({GRIPPER_OPEN_DEG:.1f} deg) -- pass "
                              f"this flag instead of editing the source if you re-measure.")
    parser.add_argument("--gripper-closed-deg", type=float, default=GRIPPER_CLOSED_DEG, dest="gripper_closed_deg",
                         help=f"Horn angle (deg) at the gripper's fully-closed end stop. Default "
                              f"{GRIPPER_CLOSED_DEG:.1f} deg -- see --gripper-open-deg.")
    parser.add_argument("--range-margin", type=float, default=GRIPPER_RANGE_MARGIN_DEG, dest="range_margin",
                         help=f"Safety margin (deg) kept between the spring center and each end stop "
                              f"(--gripper-open-deg / --gripper-closed-deg). Center gets clamped inside "
                              f"that margin. Default {GRIPPER_RANGE_MARGIN_DEG:.1f} deg.")
    parser.add_argument("--ignore-range-limit", action="store_true", dest="ignore_range_limit",
                         help="Disable the end-stop safety clamp on the spring center. Only use this if "
                              "--gripper-open-deg/--gripper-closed-deg no longer reflect the real range.")
    parser.add_argument("--temp-warning-margin", type=float, default=TEMPERATURE_WARNING_MARGIN_C,
                         dest="temp_warning_margin",
                         help=f"How many deg C below the configured overheat shutoff to start warning "
                              f"(and start checking for the overheat trip). Default "
                              f"{TEMPERATURE_WARNING_MARGIN_C:.0f} deg C.")
    parser.add_argument("--measure-freeplay", action="store_true", dest="measure_freeplay",
                         help="Diagnostic mode: torque stays OFF the whole time (no spring, no gains "
                              "applied). Just polls Present Position and tracks the running min/max "
                              "while you move the mechanism by hand -- lets you read gear backlash "
                              "directly off the horn's own encoder instead of eyeballing it. "
                              "Recommended: hold the gripper rack perfectly still, then gently rock "
                              "the horn back and forth by hand; the reported min/max range is the "
                              "backlash. All other spring-related flags are ignored in this mode.")
    parser.add_argument("--plot", action="store_true",
                         help="Show a live-updating plot (position error, velocity, acceleration, current) "
                              "while the servo runs, in addition to the text output. Requires matplotlib.")
    parser.add_argument("--plot-window", type=float, default=10.0, dest="plot_window",
                         help="How many seconds of history the live plot shows at once.")
    args = parser.parse_args()

    if args.center_at_open and args.center_at_closed:
        print("Error: --center-at-open and --center-at-closed are mutually exclusive.", file=sys.stderr)
        return 1

    try:
        args.port = resolve_port(args.port)
        return run(args)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
