import csv
from pathlib import Path

import matplotlib.pyplot as plt


session_dir = Path("../data/sessions/session_001")
trajectory_path = session_dir / "map" / "trajectory_visual.csv"
map_dir = session_dir / "map"

track_map_path = map_dir / "track_map.png"


if not trajectory_path.exists():
    print(f"Missing trajectory file: {trajectory_path}")
    raise SystemExit(1)


x_values = []
y_values = []
headings = []
times = []

with open(trajectory_path, "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        x_values.append(float(row["x"]))
        y_values.append(float(row["y"]))
        headings.append(float(row["heading"]))
        times.append(float(row["time_s"]))


if len(x_values) < 2:
    print("Not enough trajectory points to plot.")
    raise SystemExit(1)


# Normalize start position to (0, 0)
x0 = x_values[0]
y0 = y_values[0]

x_values = [x - x0 for x in x_values]
y_values = [y - y0 for y in y_values]


plt.figure(figsize=(8, 8))

plt.plot(x_values, y_values, linewidth=2, label="Visual trajectory")

# Start point
plt.scatter(x_values[0], y_values[0], s=80, label="Start")

# End point
plt.scatter(x_values[-1], y_values[-1], s=80, label="End")

plt.title("Visual Odometry Track Map")
plt.xlabel("X position (relative units)")
plt.ylabel("Y position (relative units)")
plt.axis("equal")
plt.grid(True)
plt.legend()

plt.savefig(track_map_path, dpi=200)
plt.show()

print(f"Saved track map to: {track_map_path}")
