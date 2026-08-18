"""Find a connected DYNAMIXEL servo: which port, baud rate, and ID it's on.

Run this FIRST, before spring_control.py. It tries every likely USB-serial
port against the standard DYNAMIXEL baud rates and pings IDs 0-20 using
Protocol 2.0 (which the XL330-M077 uses). When it finds a response it also
reads the model number so you can confirm it's really an XL330-M077
(model number 1190).

Usage:
    python3 scan_servo.py
    python3 scan_servo.py --port /dev/cu.usbserial-XXXX   # narrow the search
"""

from __future__ import annotations

import argparse
import glob
import sys

from dynamixel_sdk import PortHandler, PacketHandler

PROTOCOL_VERSION = 2.0
BAUD_RATES = [57600, 1000000, 115200, 9600, 200000, 250000, 400000, 500000, 2000000, 3000000, 4000000, 4500000]
ID_RANGE = range(0, 21)

XL330_M077_MODEL_NUMBER = 1190

ADDR_MODEL_NUMBER = 0


def find_candidate_ports() -> list[str]:
    patterns = [
        # macOS
        "/dev/cu.usbserial-*",
        "/dev/cu.usbmodem*",
        "/dev/cu.SLAB_USBtoUART*",
        "/dev/tty.usbserial-*",
        "/dev/tty.usbmodem*",
        # Linux (e.g. Raspberry Pi) -- U2D2 (FTDI) usually shows up as
        # ttyUSB*, some other adapters/boards as ttyACM*
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
    ]
    ports: list[str] = []
    for pattern in patterns:
        ports.extend(glob.glob(pattern))
    # de-dup while preserving order
    seen = set()
    unique = []
    for p in ports:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    # macOS creates both /dev/cu.X and /dev/tty.X for the same physical
    # device -- same adapter, two device-file names. Without this, a single
    # plugged-in adapter looks like "2 candidates" to callers. Prefer cu.*
    # (the conventional choice for outgoing/programmatic connections) and
    # drop its tty.* twin when both are present.
    cu_suffixes = {p[len("/dev/cu."):] for p in unique if p.startswith("/dev/cu.")}
    deduped = [p for p in unique if not (p.startswith("/dev/tty.") and p[len("/dev/tty."):] in cu_suffixes)]
    return deduped


def scan_port(port_name: str) -> list[tuple[str, int, int, int]]:
    """Return list of (port, baud, id, model_number) found on this port."""
    found = []
    port_handler = PortHandler(port_name)
    packet_handler = PacketHandler(PROTOCOL_VERSION)

    if not port_handler.openPort():
        print(f"  [!] Could not open {port_name}")
        return found

    try:
        for baud in BAUD_RATES:
            if not port_handler.setBaudRate(baud):
                continue
            for dxl_id in ID_RANGE:
                model_number, comm_result, error = packet_handler.ping(port_handler, dxl_id)
                if comm_result == 0:  # COMM_SUCCESS
                    print(f"  [+] Found servo on {port_name} @ {baud} baud, ID {dxl_id}, "
                          f"model number {model_number}"
                          f"{'  <-- XL330-M077' if model_number == XL330_M077_MODEL_NUMBER else ''}")
                    found.append((port_name, baud, dxl_id, model_number))
    finally:
        port_handler.closePort()

    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="Only scan this specific port (e.g. /dev/cu.usbserial-XXXX)")
    args = parser.parse_args()

    ports = [args.port] if args.port else find_candidate_ports()

    if not ports:
        print("No candidate USB-serial ports found.")
        print("Checked patterns: /dev/cu.usbserial-*, /dev/cu.usbmodem*, /dev/cu.SLAB_USBtoUART*, "
              "/dev/ttyUSB*, /dev/ttyACM*")
        print()
        print("Things to check:")
        print("  1. Is the U2D2 actually plugged into a USB port on this machine?")
        print("  2. Does it need a driver? (FTDI VCP driver, or CP210x driver depending on the")
        print("     U2D2 revision.) On macOS check System Settings > Privacy & Security if it was")
        print("     blocked; on Linux check `dmesg | tail` after plugging in, and make sure your")
        print("     user is in the `dialout` group (`sudo usermod -a -G dialout $USER`, then log out/in).")
        print("  3. Is the servo's 3-pin TTL cable connected to the U2D2 and powered (external")
        print("     power supply into the U2D2 power connector, not just USB power)?")
        print("Run `ls /dev/cu.*` and `ls /dev/tty.*` manually to see all serial devices.")
        return 1

    print(f"Scanning {len(ports)} port(s): {ports}")
    all_found = []
    for port_name in ports:
        print(f"\nScanning {port_name} ...")
        all_found.extend(scan_port(port_name))

    print()
    if not all_found:
        print("No DYNAMIXEL servo responded on any port/baud/ID combination.")
        print("Double check power to the servo and the TTL cable orientation/seating.")
        return 1

    print(f"Found {len(all_found)} servo(s). Use the port/baud/ID above with spring_control.py, e.g.:")
    port, baud, dxl_id, _ = all_found[0]
    print(f"  python3 spring_control.py --port {port} --baud {baud} --id {dxl_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
