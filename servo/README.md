# XL330-M077 Servo-as-Spring

Makes a physical Dynamixel XL330-M077 servo behave like a torsion spring:
twist it away from a center point and it pushes back proportionally, then
returns when released.

## Hardware setup (this machine)

- Servo: Dynamixel XL330-M077 (model number 1190), Protocol 2.0
- Interface: ROBOTIS U2D2 USB adapter
- Port: `/dev/cu.usbserial-FTBEQKWH`
- Baud: 57600
- ID: 1

Run `scan_servo.py` any time to re-detect these (e.g. if the U2D2 gets
plugged into a different USB port and the device name changes).

## Files

| File | What it is |
|---|---|
| `scan_servo.py` | Finds the servo's port/baud/ID by pinging across the standard options. Run this first if the servo isn't found. |
| `spring_position_mode.py` | **The working spring controller. Use this one.** |
| `spring_control.py` | An earlier, external-current-control approach. Kept for reference, but it is **not reliably stable** on this hardware (see below) — do not use as the default. |
| `spirng-simulation.py` | The original pure-math simulation (no hardware, just plots spring-mass-damper behavior). Unrelated to driving the real servo. |

## Quick start

```bash
cd "/Users/thiemchen/Desktop/hardware project/servo"
python3 spring_position_mode.py
```

Twist the servo horn and let go — it should spring back and settle cleanly.
`Ctrl+C` stops it and disables torque.

Tuning flags:

- `--p-gain` (default 400, factory value) — higher = stiffer/snappier spring
- `--d-gain` (default 800) — higher = more damped, less overshoot on release
- `--max-current` (default 150 mA) — torque cap; also a safety limit
- `--center` (default 0) — spring center offset in degrees from wherever the
  horn is when the script starts
- `--duration` — run for N seconds instead of until Ctrl+C

## How it works

`spring_position_mode.py` puts the servo in **Current-based Position
Control Mode** (Operating Mode 5):

1. Set Position P/I/D gain registers (stiffness/damping "feel").
2. Set Goal Current as a torque cap (how hard it can push back).
3. Set Goal Position to the center point, **once**.

From there, the servo's own onboard microcontroller runs the position PID
loop and holds/restores that position on its own, at its own fast internal
update rate. The Python script just polls `Present Position` occasionally
(10 Hz) to print status — that polling rate has no bearing on stability.

## Why not just compute the spring force from the PC? (`spring_control.py`)

The first approach read the servo's position over USB every control cycle,
computed `current = -stiffness * error - damping * velocity` on the PC, and
wrote that back as `Goal Current` (raw Current Control Mode). This is the
more "obvious" way to implement a virtual spring, and it's worth recording
why it didn't work out here.

**Root cause: USB latency.** This SDK assumes ~16 ms of USB latency per
serial transaction on this kind of FTDI/U2D2 link (`LATENCY_TIMER` in
`dynamixel_sdk/port_handler.py`). Even after combining position+velocity+
current into a single read transaction, the achieved loop rate measured out
to **~31 Hz (~32 ms round trip)**, not the 100 Hz requested. Every force
update was therefore based on already-stale position data.

That delay turned out to be enough, by itself, to destabilize a naive
proportional-plus-damping virtual spring. Across this session, several
tuning attempts were tried and physically tested on the real servo, and
each one failed in a different, instructive way:

| Attempt | Result |
|---|---|
| `stiffness=1.5`, `damping=0.2`, no filtering | Seemed stable hands-off; a real nudge grew into a fast, sustained "bang-bang" chatter (current flipping full-scale every single sample) |
| `damping` raised to `0.8` to add more settling | Made the fast chatter *worse*, not better — a delayed damping term at that lag was adding energy instead of removing it |
| `damping=0` + low-pass filter on output current (to avoid the delayed-derivative problem) | A pure delayed spring (no damping at all) is not stable either — a light nudge grew into an *unbounded* multi-turn spin |
| Reintroduced modest damping (`~0.25`) + light filter + a tight slew-rate limiter as a safety backstop | The slew-rate limiter itself became a relay-like nonlinearity: after one nudge, the commanded current locked into a perfectly repeating `-110 / -50 / +50 / +110 mA` cycle, forever, and never decayed |
| Loosened the slew limiter back off | Reintroduced fast, chaotic chattering |
| Stripped back down to plain PD (`stiffness=1.5`, `damping=0.1`), no filter, no slew limit | Still developed a clean, slowly-growing sustained oscillation after a nudge |

The consistent finding: **the ~32 ms round-trip delay itself was the
problem**, not the specific gain values. No PD tuning found in this session
avoided some form of self-sustained oscillation once physically nudged,
even though several configurations looked fine when just sitting still
untouched.

`spring_position_mode.py` sidesteps this whole class of problem by moving
the closed loop onto the servo itself, where there's no USB round trip in
the control path at all. It was tested the same way (hands-off, then a live
session including hard nudges and holding it away from center) and settled
cleanly every time.

## Safety notes

- Keep fingers clear of pinch points while torque is enabled.
- Start with a low `--max-current` if attaching a different/heavier horn or
  load than what was tested here (bare horn, no attached mechanism).
- `Ctrl+C` always disables torque before the script exits.
