"""Simulate a Dynamixel XL-330 servo as a torsional spring-damper system.

This script does not control the physical servo. It models the servo's angular
behavior with a spring-like restoring torque against position error.
"""

from __future__ import annotations

import math
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:  # pragma: no cover
    plt = None
    np = None


class ServoSpringModel:
    """A simple servo model using spring, damper, and inertia."""

    def __init__(
        self,
        stiffness_nm_per_rad: float = 1.0,
        damping_nm_per_rad_s: float = 0.05,
        inertia_kg_m2: float = 2e-5,
        max_torque_nm: float = 1.0,
        min_angle_deg: float = -150.0,
        max_angle_deg: float = 150.0,
    ) -> None:
        self.stiffness = stiffness_nm_per_rad
        self.damping = damping_nm_per_rad_s
        self.inertia = inertia_kg_m2
        self.max_torque = max_torque_nm
        self.min_angle = math.radians(min_angle_deg)
        self.max_angle = math.radians(max_angle_deg)

        self.angle = 0.0
        self.angular_velocity = 0.0

    def reset(self, angle_deg: float = 0.0, velocity_deg_s: float = 0.0) -> None:
        self.angle = math.radians(angle_deg)
        self.angular_velocity = math.radians(velocity_deg_s)

    def simulate_step(self, target_angle_deg: float, dt: float) -> float:
        target_angle = math.radians(target_angle_deg)
        error = target_angle - self.angle
        spring_torque = self.stiffness * error
        damping_torque = -self.damping * self.angular_velocity

        torque = spring_torque + damping_torque
        torque = max(min(torque, self.max_torque), -self.max_torque)

        angular_acceleration = torque / self.inertia
        self.angular_velocity += angular_acceleration * dt
        self.angle += self.angular_velocity * dt

        if self.angle < self.min_angle:
            self.angle = self.min_angle
            self.angular_velocity = 0.0
        elif self.angle > self.max_angle:
            self.angle = self.max_angle
            self.angular_velocity = 0.0

        return math.degrees(self.angle)

    def current_torque(self, target_angle_deg: float) -> float:
        target_angle = math.radians(target_angle_deg)
        error = target_angle - self.angle
        torque = self.stiffness * error - self.damping * self.angular_velocity
        return max(min(torque, self.max_torque), -self.max_torque)


def run_simulation(model: ServoSpringModel, target_profile: list[float], dt: float) -> dict[str, list[float]]:
    angles: list[float] = []
    torques: list[float] = []
    times: list[float] = []

    time = 0.0
    for target in target_profile:
        angle = model.simulate_step(target, dt)
        torque = model.current_torque(target)
        angles.append(angle)
        torques.append(torque)
        times.append(time)
        time += dt

    return {
        "time": times,
        "angle": angles,
        "torque": torques,
        "target": target_profile,
    }


def make_profile(step_time: float, dt: float, initial: float, step: float, duration: float) -> list[float]:
    profile: list[float] = []
    steps = int(duration / dt)
    switch_at = int(step_time / dt)
    for i in range(steps):
        profile.append(initial if i < switch_at else step)
    return profile


def plot_results(results: dict[str, list[float]]) -> None:
    if plt is None or np is None:
        print("Matplotlib or NumPy is not installed. Install them with `pip install matplotlib numpy`." )
        return

    time = np.array(results["time"])
    angle = np.array(results["angle"])
    target = np.array(results["target"])
    torque = np.array(results["torque"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(time, angle, label="Actual angle")
    ax1.plot(time, target, "--", label="Target angle")
    ax1.set_ylabel("Angle (deg)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time, torque, color="tab:red", label="Torque")
    ax2.set_ylabel("Torque (N·m)")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def print_summary(results: dict[str, list[float]]) -> None:
    final_angle = results["angle"][-1]
    final_target = results["target"][-1]
    final_torque = results["torque"][-1]

    print("Simulation summary")
    print("------------------")
    print(f"Final actual angle: {final_angle:.2f} deg")
    print(f"Final target angle: {final_target:.2f} deg")
    print(f"Final torque estimate: {final_torque:.3f} N·m")


def main() -> int:
    dt = 0.002
    model = ServoSpringModel(
        stiffness_nm_per_rad=1.5,
        damping_nm_per_rad_s=0.12,
        inertia_kg_m2=2.5e-5,
        max_torque_nm=1.0,
    )
    model.reset(angle_deg=0.0)

    target_profile = make_profile(step_time=0.5, dt=dt, initial=0.0, step=90.0, duration=2.5)
    results = run_simulation(model, target_profile, dt)

    print_summary(results)
    plot_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
