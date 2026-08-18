"""Make a physical XL330-M077 behave like a torsional spring in real time.

This drives the REAL servo (not a math-only simulation) using DYNAMIXEL
Current Control Mode: every control cycle it reads the present angle and
velocity in one transaction and writes
goal_current = -stiffness*(angle-center) - damping*velocity, clamped to
--max-current. The result is a servo that resists being twisted away from
`center` and springs back when released, like a torsion spring.

This is deliberately the simplest version of this loop. Several "smarter"
additions were tried in this session and each one made things worse instead
of better, so they were removed rather than kept as options:
  - damping=0 (pure spring, no velocity term): a light nudge grew into an
    unbounded, multi-turn spin. Some damping is required, not optional.
  - a low-pass filter on the output current, added to avoid delay issues:
    with damping still at 0 it didn't help (see above); once damping was
    reintroduced the filter's own lag started contributing to slow,
    sustained oscillation.
  - a slew-rate limiter, added as a safety backstop: instead it became a
    relay-like nonlinearity that locked the system into a clean, perfectly
    repeating, never-decaying limit cycle (measured: commanded current
    cycling through exactly -110/-50/+50/+110 mA forever after one nudge).
    Loosening it back off just reintroduced fast chattering instead.
A modest --damping with no filter and no slew limiting is what actually
held steady hands-off AND settled after a nudge in this session's testing.
Keep --damping well above 0; don't reach for filtering/slew-limiting to
paper over instability -- lower --stiffness and/or raise --damping instead.

Run scan_servo.py first to confirm the port/baud/ID (already found in this
setup: /dev/cu.usbserial-FTBEQKWH, 57600 baud, ID 1).

Safety notes:
  - Start with a low --stiffness and --max-current and increase gradually.
    High current + high stiffness can make the arm snap back hard.
  - Keep fingers clear of pinch points while torque is enabled.
  - Ctrl+C always disables torque before exiting.

Usage:
    python3 spring_control.py --port /dev/cu.usbserial-FTBEQKWH
    python3 spring_control.py --stiffness 2 --damping 0.4 --max-current 200
"""

from __future__ import annotations

import argparse
import struct
import sys
import time

from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

PROTOCOL_VERSION = 2.0

# XL330-M077 control table (Protocol 2.0 X-series layout).
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_CURRENT = 102
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132

# Present Current (2B) + Present Velocity (4B) + Present Position (4B) are
# back-to-back registers (126..135), so they can be read in a single serial
# transaction instead of three. That matters a lot here: this SDK assumes
# ~16ms of USB latency per transaction (see LATENCY_TIMER in port_handler.py),
# so every extra round trip directly adds delay between "sense" and "push
# back" -- and a delayed restoring force is exactly what makes a virtual
# spring ring/oscillate on its own, with no hand on it.
ADDR_STATE_BLOCK = ADDR_PRESENT_CURRENT
STATE_BLOCK_LEN = 10

OPERATING_MODE_CURRENT_CONTROL = 0

POSITION_UNIT_DEG = 0.088          # deg per position tick (4096 ticks/rev)
VELOCITY_UNIT_DEG_S = 0.229 * 6.0  # 0.229 rev/min per tick -> deg/s per tick
CURRENT_UNIT_MA = 1.0              # mA per Goal/Present Current tick (XL330 series)


def try_reduce_usb_latency(port_handler) -> bool:
    """Best-effort: ask the macOS FTDI driver for ~1ms latency instead of the
    default 16ms. Silently does nothing if unsupported (non-macOS, non-FTDI,
    or the ioctl isn't available) -- this is an optimization, not a
    requirement, so failure here should never break the connection."""
    if sys.platform != "darwin":
        return False
    try:
        import fcntl

        IOSSDATALAT = 0x80045437  # macOS IOKit ioctl: set FTDI latency timer, in microseconds
        buf = struct.pack("L", 1000)  # request 1ms
        fcntl.ioctl(port_handler.ser.fileno(), IOSSDATALAT, buf)
        return True
    except Exception:
        return False


def to_signed(value: int, bits: int) -> int:
    mask = 1 << (bits - 1)
    return (value ^ mask) - mask


def to_unsigned(value: int, bits: int) -> int:
    return value & ((1 << bits) - 1)


class DynamixelSpring:
    def __init__(self, port_name: str, baud: int, dxl_id: int, max_current_ma: float):
        self.dxl_id = dxl_id
        self.max_current_ma = max_current_ma

        self.port_handler = PortHandler(port_name)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)

        if not self.port_handler.openPort():
            raise RuntimeError(f"Could not open port {port_name}")
        if not self.port_handler.setBaudRate(baud):
            raise RuntimeError(f"Could not set baud rate {baud} on {port_name}")

        if try_reduce_usb_latency(self.port_handler):
            print("Reduced USB latency timer to ~1ms (macOS FTDI ioctl).")
        else:
            print("Could not reduce USB latency timer (non-fatal); loop rate may be limited "
                  "by the default ~16ms FTDI latency.")

        model_number, comm_result, error = self.packet_handler.ping(self.port_handler, dxl_id)
        if comm_result != COMM_SUCCESS:
            raise RuntimeError(
                f"Could not ping servo ID {dxl_id} on {port_name} @ {baud}. "
                f"Run scan_servo.py to re-check the port/baud/ID."
            )
        print(f"Connected: ID {dxl_id}, model number {model_number}")

        self._set_torque(False)
        self._write1(ADDR_OPERATING_MODE, OPERATING_MODE_CURRENT_CONTROL)
        self._set_torque(True)

    def _check(self, comm_result: int, error: int, what: str) -> None:
        if comm_result != COMM_SUCCESS:
            raise RuntimeError(f"{what} failed: {self.packet_handler.getTxRxResult(comm_result)}")
        if error != 0:
            raise RuntimeError(f"{what} reported hardware error: {self.packet_handler.getRxPacketError(error)}")

    def _write1(self, addr: int, value: int) -> None:
        result, error = self.packet_handler.write1ByteTxRx(self.port_handler, self.dxl_id, addr, value)
        self._check(result, error, f"write1 addr={addr}")

    def _set_torque(self, enable: bool) -> None:
        self._write1(ADDR_TORQUE_ENABLE, 1 if enable else 0)

    def read_position_deg(self) -> float:
        raw, result, error = self.packet_handler.read4ByteTxRx(self.port_handler, self.dxl_id, ADDR_PRESENT_POSITION)
        self._check(result, error, "read present position")
        return to_signed(raw, 32) * POSITION_UNIT_DEG

    def read_velocity_deg_s(self) -> float:
        raw, result, error = self.packet_handler.read4ByteTxRx(self.port_handler, self.dxl_id, ADDR_PRESENT_VELOCITY)
        self._check(result, error, "read present velocity")
        return to_signed(raw, 32) * VELOCITY_UNIT_DEG_S

    def read_current_ma(self) -> float:
        raw, result, error = self.packet_handler.read2ByteTxRx(self.port_handler, self.dxl_id, ADDR_PRESENT_CURRENT)
        self._check(result, error, "read present current")
        return to_signed(raw, 16) * CURRENT_UNIT_MA

    def read_state(self) -> tuple[float, float, float]:
        """One serial transaction instead of three: returns
        (current_ma, velocity_deg_s, position_deg)."""
        data, result, error = self.packet_handler.readTxRx(self.port_handler, self.dxl_id, ADDR_STATE_BLOCK, STATE_BLOCK_LEN)
        self._check(result, error, "read state block")
        current_raw, velocity_raw, position_raw = struct.unpack("<hii", bytes(data))
        return (
            current_raw * CURRENT_UNIT_MA,
            velocity_raw * VELOCITY_UNIT_DEG_S,
            position_raw * POSITION_UNIT_DEG,
        )

    def write_goal_current_ma(self, current_ma: float) -> float:
        """Write the (clamped) goal current and return the value actually sent."""
        clipped = max(min(current_ma, self.max_current_ma), -self.max_current_ma)
        raw = to_unsigned(int(round(clipped / CURRENT_UNIT_MA)), 16)
        result, error = self.packet_handler.write2ByteTxRx(self.port_handler, self.dxl_id, ADDR_GOAL_CURRENT, raw)
        self._check(result, error, "write goal current")
        return clipped

    def close(self) -> None:
        try:
            self.write_goal_current_ma(0.0)
            self._set_torque(False)
        except Exception:
            pass
        self.port_handler.closePort()


def run_spring(args: argparse.Namespace) -> int:
    servo = DynamixelSpring(args.port, args.baud, args.id, args.max_current)

    try:
        start_angle = servo.read_position_deg()
        center = start_angle + args.center
        print(f"Starting angle: {start_angle:.2f} deg")
        print(f"Spring center:  {center:.2f} deg")
        print(f"Stiffness: {args.stiffness} mA/deg, Damping: {args.damping} mA/(deg/s), "
              f"Max current: {args.max_current} mA")
        print("Torque is now enabled. Twist the horn to feel the spring. Ctrl+C to stop.\n")

        period = 1.0 / args.rate
        start_time = time.monotonic()
        next_print = start_time
        loop_count = 0
        rate_window_start = start_time
        rate_window_count = 0
        achieved_hz = 0.0

        while args.duration is None or (time.monotonic() - start_time) < args.duration:
            loop_start = time.monotonic()

            _current_ma, velocity, angle = servo.read_state()

            error = angle - center
            # Plain PD: proportional restoring force + small damping. Extra
            # machinery (output low-pass filter, slew-rate limiter) was tried
            # here and, empirically, each one added its own failure mode
            # instead of fixing the underlying one:
            #   - filter alone (no damping): undamped delayed spring, a light
            #     nudge grew into an unbounded multi-turn spin.
            #   - tight slew limit + filter + damping: the rate limiter itself
            #     became a relay-like nonlinearity that locked into a clean,
            #     perfectly repeating, never-decaying limit cycle (measured:
            #     current cycling exactly -110/-50/+50/+110 mA forever).
            #   - loosening the slew limit back off: reintroduced fast,
            #     chaotic chattering.
            # Keeping the loop this simple, with a single combined
            # position+velocity+current read per cycle (see read_state) and a
            # small, empirically-verified damping value, is what actually
            # stayed stable across a hands-off test and a nudge-and-release
            # test (see spring_test_log*.txt analyses in this session).
            requested_current = -args.stiffness * error - args.damping * velocity
            applied_current = servo.write_goal_current_ma(requested_current)

            loop_count += 1
            rate_window_count += 1
            if loop_start - rate_window_start >= 1.0:
                achieved_hz = rate_window_count / (loop_start - rate_window_start)
                rate_window_start = loop_start
                rate_window_count = 0

            if loop_start >= next_print:
                clamped_flag = " (clamped)" if applied_current != requested_current else ""
                print(f"\rangle={angle:7.2f} deg  error={error:7.2f} deg  "
                      f"cmd_current={applied_current:7.1f} mA{clamped_flag}  "
                      f"loop={achieved_hz:5.1f} Hz (target {args.rate:.0f} Hz)   ", end="", flush=True)
                next_print = loop_start + 0.1

            elapsed = time.monotonic() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if achieved_hz and achieved_hz < 0.5 * args.rate:
            print(f"\n\nNote: achieved loop rate ({achieved_hz:.1f} Hz) is well below the "
                  f"target ({args.rate:.0f} Hz). If it still oscillates, try a lower "
                  f"--stiffness and/or a higher --damping.")

    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C).")
    finally:
        servo.close()
        print("Torque disabled, port closed.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/cu.usbserial-FTBEQKWH", help="Serial port for the U2D2")
    parser.add_argument("--baud", type=int, default=57600, help="Baud rate")
    parser.add_argument("--id", type=int, default=1, help="DYNAMIXEL ID")
    parser.add_argument("--stiffness", type=float, default=1.5, help="Spring stiffness, mA of current per degree of error")
    parser.add_argument("--damping", type=float, default=0.1,
                         help="Damping, mA per deg/s of velocity. Load-bearing, not optional: damping=0 lets a "
                              "light nudge grow into an unbounded oscillation (reproduced this session). Don't "
                              "zero this out. If it still doesn't settle after a nudge, raise this a bit before "
                              "reaching for anything fancier -- filtering/slew-limiting were tried here and each "
                              "one introduced its own oscillation instead of fixing this (see module docstring).")
    parser.add_argument("--max-current", type=float, default=150.0, dest="max_current", help="Current clamp, mA (safety limit)")
    parser.add_argument("--center", type=float, default=0.0, help="Spring center offset in degrees from the starting position")
    parser.add_argument("--rate", type=float, default=100.0, help="Control loop rate, Hz")
    parser.add_argument("--duration", type=float, default=None, help="Run time in seconds (default: run until Ctrl+C)")
    args = parser.parse_args()

    try:
        return run_spring(args)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
