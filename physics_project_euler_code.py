import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


G = 9.81
RHO = 1.225
DT = 0.0005

OUTPUT_DIR = Path("physics_project_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

BALLS = {
    "Table tennis": {"m": 0.0027, "d": 0.040, "Cd": 0.50},
    "Tennis": {"m": 0.0570, "d": 0.067, "Cd": 0.50},
    "Golf": {"m": 0.0459, "d": 0.043, "Cd": 0.25},
    "Cricket": {"m": 0.1560, "d": 0.072, "Cd": 0.45},
    "Football": {"m": 0.4300, "d": 0.220, "Cd": 0.25},
    "Shot put": {"m": 7.2600, "d": 0.110, "Cd": 0.47},
}

for ball in BALLS.values():
    ball["r"] = ball["d"] / 2
    ball["A"] = math.pi * ball["r"] ** 2


def lift_coefficient(speed: float, radius: float, omega: float) -> float:
    """Empirical lift coefficient used in the project."""
    if abs(omega) < 1e-12 or speed < 1e-12:
        return 0.0
    spin_parameter = abs(radius * omega / speed)
    return 1.0 / (2.0 + 1.0 / spin_parameter)


def euler_2d(
    ball: dict,
    launch_speed: float,
    angle_deg: float,
    release_height: float = 0.0,
    omega: float = 0.0,
    include_drag: bool = True,
    include_magnus: bool = True,
    max_time: float = 30.0,
):
    #Calculate a two-dimensional trajectory using the explicit Euler method.
    theta = math.radians(angle_deg)

    x = 0.0
    y = release_height
    vx = launch_speed * math.cos(theta)
    vy = launch_speed * math.sin(theta)

    x_values = [x]
    y_values = [y]

    time = 0.0

    while time < max_time:
        speed = math.hypot(vx, vy)

        ax = 0.0
        ay = -G

        if include_drag and speed > 0:
            drag_factor = (
                0.5 * RHO * ball["Cd"] * ball["A"] / ball["m"]
            )
            ax -= drag_factor * speed * vx
            ay -= drag_factor * speed * vy

        if include_magnus and abs(omega) > 0 and speed > 0:
            cl = lift_coefficient(speed, ball["r"], omega)
            magnus_factor = 0.5 * RHO * cl * ball["A"] / ball["m"]
            direction = 1 if omega > 0 else -1

            ax += direction * magnus_factor * speed * (-vy)
            ay += direction * magnus_factor * speed * vx

        x_new = x + vx * DT
        y_new = y + vy * DT
        vx_new = vx + ax * DT
        vy_new = vy + ay * DT

        time += DT

        if y_new < 0 and time > DT:
            fraction = y / (y - y_new)
            landing_x = x + fraction * (x_new - x)
            x_values.append(landing_x)
            y_values.append(0.0)
            break

        x, y = x_new, y_new
        vx, vy = vx_new, vy_new

        x_values.append(x)
        y_values.append(y)

    return np.array(x_values), np.array(y_values)


def euler_3d_free_kick(
    ball: dict,
    launch_speed: float,
    angle_deg: float,
    omega_z: float = 0.0,
    max_time: float = 10.0,
):
    #Calculate a three-dimensional football trajectory with sidespin.
    theta = math.radians(angle_deg)

    x = y = z = 0.0
    vx = launch_speed * math.cos(theta)
    vy = 0.0
    vz = launch_speed * math.sin(theta)

    xs, ys, zs = [x], [y], [z]
    time = 0.0

    while time < max_time:
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)

        drag_factor = 0.5 * RHO * ball["Cd"] * ball["A"] / ball["m"]

        ax = -drag_factor * speed * vx
        ay = -drag_factor * speed * vy
        az = -G - drag_factor * speed * vz

        if omega_z and speed > 0:
            cl = lift_coefficient(speed, ball["r"], omega_z)
            magnus_factor = 0.5 * RHO * cl * ball["A"] / ball["m"]

            ay += magnus_factor * speed * vx
            ax -= magnus_factor * speed * vy

        x_new = x + vx * DT
        y_new = y + vy * DT
        z_new = z + vz * DT

        vx_new = vx + ax * DT
        vy_new = vy + ay * DT
        vz_new = vz + az * DT

        time += DT

        if z_new < 0 and time > DT:
            fraction = z / (z - z_new)
            xs.append(x + fraction * (x_new - x))
            ys.append(y + fraction * (y_new - y))
            zs.append(0.0)
            break

        x, y, z = x_new, y_new, z_new
        vx, vy, vz = vx_new, vy_new, vz_new

        xs.append(x)
        ys.append(y)
        zs.append(z)

    return np.array(xs), np.array(ys), np.array(zs)


def terminal_velocity(ball: dict) -> float:
    return math.sqrt(
        2 * ball["m"] * G / (RHO * ball["Cd"] * ball["A"])
    )


def ball_exit_speed(
    bat_mass: float,
    restitution: float,
    ball_mass: float = 0.156,
    incoming_ball_speed: float = 35.0,
    bat_speed: float = 15.0,
) -> float:
    """Ball speed after a one-dimensional bat-ball collision."""
    numerator = (
        (ball_mass - restitution * bat_mass) * (-incoming_ball_speed)
        + bat_mass * (1 + restitution) * bat_speed
    )
    return numerator / (bat_mass + ball_mass)


def save_figure(filename: str):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    cricket = BALLS["Cricket"]
    tennis = BALLS["Tennis"]
    football = BALLS["Football"]
    shot_put = BALLS["Shot put"]

    # Figure 1
    plt.figure(figsize=(6.5, 4.0))
    for angle in [15, 30, 45, 60, 75]:
        x, y = euler_2d(
            cricket, 25, angle,
            include_drag=False,
            include_magnus=False,
        )
        plt.plot(x, y, label=f"{angle}°")
    plt.title("Fig. 1  Trajectories in vacuum, u = 25 m/s")
    plt.xlabel("Horizontal distance x (m)")
    plt.ylabel("Height y (m)")
    plt.legend(title="Launch angle")
    save_figure("figure_1.png")

    # Figure 2
    angles = np.arange(1, 90, 0.5)
    ranges = []
    for angle in angles:
        x, _ = euler_2d(
            cricket, 25, float(angle),
            include_drag=False,
            include_magnus=False,
        )
        ranges.append(x[-1])

    plt.figure(figsize=(6.5, 4.0))
    plt.plot(angles, ranges)
    plt.scatter([45], [ranges[np.argmin(abs(angles - 45))]])
    plt.title("Fig. 2  Range against launch angle in vacuum")
    plt.xlabel("Launch angle θ (degrees)")
    plt.ylabel("Range R (m)")
    save_figure("figure_2.png")

    # Figure 3
    x_vac, y_vac = euler_2d(
        cricket, 35, 45,
        include_drag=False,
        include_magnus=False,
    )
    x_air, y_air = euler_2d(
        cricket, 35, 45,
        include_drag=True,
        include_magnus=False,
    )

    plt.figure(figsize=(6.5, 4.0))
    plt.plot(x_vac, y_vac, label=f"Vacuum (R = {x_vac[-1]:.1f} m)")
    plt.plot(x_air, y_air, label=f"With air drag (R = {x_air[-1]:.1f} m)")
    plt.title("Fig. 3  Effect of air drag on a cricket ball")
    plt.xlabel("Horizontal distance x (m)")
    plt.ylabel("Height y (m)")
    plt.legend()
    save_figure("figure_3.png")

    # Figure 4 and Table 2
    fine_angles = np.arange(10, 80.01, 0.25)
    vacuum_ranges = []
    air_ranges = []

    for angle in fine_angles:
        x1, _ = euler_2d(
            cricket, 35, float(angle),
            include_drag=False,
            include_magnus=False,
        )
        x2, _ = euler_2d(
            cricket, 35, float(angle),
            include_drag=True,
            include_magnus=False,
        )
        vacuum_ranges.append(x1[-1])
        air_ranges.append(x2[-1])

    vacuum_ranges = np.array(vacuum_ranges)
    air_ranges = np.array(air_ranges)
    best_angle = fine_angles[np.argmax(air_ranges)]

    plt.figure(figsize=(6.5, 4.0))
    plt.plot(fine_angles, vacuum_ranges, label="Vacuum")
    plt.plot(fine_angles, air_ranges, label="With air drag")
    plt.axvline(45, linestyle="--", linewidth=1)
    plt.axvline(best_angle, linestyle="--", linewidth=1)
    plt.title("Fig. 4  Range against launch angle")
    plt.xlabel("Launch angle θ (degrees)")
    plt.ylabel("Range R (m)")
    plt.legend()
    save_figure("figure_4.png")

    # Figure 5 and Table 3
    plt.figure(figsize=(6.5, 4.0))
    spin_ranges = {}

    for label, omega in [
        ("Backspin", 300),
        ("No spin", 0),
        ("Topspin", -300),
    ]:
        x, y = euler_2d(
            tennis, 30, 16,
            omega=omega,
            include_drag=True,
            include_magnus=True,
        )
        spin_ranges[label] = x[-1]
        plt.plot(x, y, label=f"{label} (R = {x[-1]:.1f} m)")

    plt.title("Fig. 5  Magnus effect on a tennis ball")
    plt.xlabel("Horizontal distance x (m)")
    plt.ylabel("Height y (m)")
    plt.legend()
    save_figure("figure_5.png")

    # Figure 6
    x0, y0, z0 = euler_3d_free_kick(football, 28, 12, 0)
    xs, ys, zs = euler_3d_free_kick(football, 28, 12, 60)

    figure, axes = plt.subplots(2, 1, figsize=(6.5, 5.4), sharex=True)
    axes[0].plot(x0, y0, label="No spin")
    axes[0].plot(xs, ys, label="Sidespin 60 rad/s")
    axes[0].set_ylabel("Lateral deflection y (m)")
    axes[0].legend()
    axes[0].set_title("Fig. 6  Curled free kick")

    axes[1].plot(x0, z0, label="No spin")
    axes[1].plot(xs, zs, label="Sidespin")
    axes[1].axhline(2.44, linestyle="--", linewidth=1, label="Crossbar")
    axes[1].set_xlabel("Distance down the pitch x (m)")
    axes[1].set_ylabel("Height z (m)")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "figure_6.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    # Figure 7
    speeds = np.linspace(0, 45, 150)

    plt.figure(figsize=(6.5, 4.0))
    for name in ["Table tennis", "Tennis", "Golf", "Cricket", "Football"]:
        ball = BALLS[name]
        drag_force = 0.5 * RHO * ball["Cd"] * ball["A"] * speeds ** 2
        plt.plot(speeds, drag_force, label=name)

    plt.title("Fig. 7  Drag force grows as the square of speed")
    plt.xlabel("Speed v (m/s)")
    plt.ylabel("Drag force F_D (N)")
    plt.legend()
    save_figure("figure_7.png")

    # Figure 8 and Table 1
    names = list(BALLS.keys())
    velocities = [terminal_velocity(BALLS[name]) for name in names]

    plt.figure(figsize=(6.5, 4.0))
    bars = plt.bar(names, velocities)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Terminal velocity v_T (m/s)")
    plt.title("Fig. 8  Terminal velocity of sports objects")

    for bar, value in zip(bars, velocities):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.0f}",
            ha="center",
            fontsize=8,
        )

    save_figure("figure_8.png")

    # Figure 9 and Table 4
    release_angles = np.arange(25, 55.01, 0.1)

    plt.figure(figsize=(6.5, 4.0))
    for height in [0, 1, 2]:
        distances = []
        for angle in release_angles:
            x, _ = euler_2d(
                shot_put, 13.5, float(angle),
                release_height=height,
                include_drag=False,
                include_magnus=False,
            )
            distances.append(x[-1])

        distances = np.array(distances)
        index = np.argmax(distances)

        plt.plot(
            release_angles,
            distances,
            label=f"Release height {height} m",
        )
        plt.scatter(
            [release_angles[index]],
            [distances[index]],
            s=20,
        )

    plt.title("Fig. 9  Effect of release height in shot put")
    plt.xlabel("Release angle θ (degrees)")
    plt.ylabel("Distance thrown R (m)")
    plt.legend()
    save_figure("figure_9.png")

    # Figure 10
    bat_masses = np.linspace(0.35, 3.0, 200)

    plt.figure(figsize=(6.5, 4.0))
    for e in [0.3, 0.4, 0.5, 0.6]:
        values = [ball_exit_speed(mass, e) for mass in bat_masses]
        plt.plot(bat_masses, values, label=f"e = {e}")

    plt.axvline(1.2, linestyle="--", linewidth=1)
    plt.title("Fig. 10  Ball exit speed against bat mass")
    plt.xlabel("Mass of bat M (kg)")
    plt.ylabel("Speed of ball after impact (m/s)")
    plt.legend()
    save_figure("figure_10.png")

    # Figure 11
    run_up_speeds = np.linspace(6, 11, 150)
    rise = run_up_speeds ** 2 / (2 * G)
    estimated_bar_height = rise + 1.0

    plt.figure(figsize=(6.5, 4.0))
    plt.plot(run_up_speeds, rise, label="Rise of centre of mass")
    plt.plot(
        run_up_speeds,
        estimated_bar_height,
        linestyle="--",
        label="Estimated bar height",
    )
    plt.scatter(
        [9.5],
        [9.5 ** 2 / (2 * G) + 1.0],
        s=25,
    )
    plt.title("Fig. 11  Pole vault energy model")
    plt.xlabel("Run-up speed v (m/s)")
    plt.ylabel("Height (m)")
    plt.legend()
    save_figure("figure_11.png")

    # Print the principal numerical results
    print(f"Best cricket-ball angle with drag: {best_angle:.1f}°")
    print(f"Cricket range at 45° in vacuum: {x_vac[-1]:.2f} m")
    print(f"Cricket range at 45° with drag: {x_air[-1]:.2f} m")
    print(f"Free-kick sideways deflection: {ys[-1]:.2f} m")
    print("Tennis-ball ranges:")
    for label, value in spin_ranges.items():
        print(f"  {label}: {value:.2f} m")


if __name__ == "__main__":
    main()
