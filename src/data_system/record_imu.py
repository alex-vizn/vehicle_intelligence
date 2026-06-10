import time
import csv
import json
import signal
from pathlib import Path

import board
import busio

from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_ROTATION_VECTOR,
    BNO_REPORT_GRAVITY,
    BNO_REPORT_LINEAR_ACCELERATION,
)
from adafruit_bno08x.i2c import BNO08X_I2C


# ---------------- CONFIG ----------------

TARGET_HZ = 100
SAMPLE_DELAY = 1.0 / TARGET_HZ

AXIS_CONVENTION = {
    "x": "forward",
    "y": "right",
    "z": "up"
}

running = True


def handle_stop(signum, frame):
    global running
    running = False


signal.signal(signal.SIGINT, handle_stop)
signal.signal(signal.SIGTERM, handle_stop)


# ---------------- HELPERS ----------------

def make_output_file(session_path: Path):
    raw_dir = session_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    session_name = session_path.name
    return raw_dir / f"{session_name}_imu.csv"


def update_meta(session_path: Path, imu_file: Path):
    meta_file = session_path / "meta.json"

    if not meta_file.exists():
        return

    try:
        meta = json.loads(meta_file.read_text())
        meta["imu_file"] = f"raw/{imu_file.name}"
        meta["imu_model"] = "BNO085"
        meta["imu_target_hz"] = TARGET_HZ
        meta["imu_axis_convention"] = AXIS_CONVENTION
        meta["imu_started_at_ns"] = time.time_ns()
        meta_file.write_text(json.dumps(meta, indent=2))
    except Exception as e:
        print(f"WARNING: could not update meta.json: {e}")


# ---------------- MAIN ----------------

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 record_imu.py <session_path>")
        return

    session_path = Path(sys.argv[1])
    imu_file = make_output_file(session_path)

    print(f"IMU logging to: {imu_file}")

    # I2C setup
    i2c = busio.I2C(board.SCL, board.SDA)
    bno = BNO08X_I2C(i2c)

    # Enable reports
    bno.enable_feature(BNO_REPORT_ACCELEROMETER)
    bno.enable_feature(BNO_REPORT_GYROSCOPE)
    bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
    bno.enable_feature(BNO_REPORT_GRAVITY)
    bno.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)

    update_meta(session_path, imu_file)

    fieldnames = [
        "time_ns",
        "time_s",

        # Raw acceleration, m/s^2
        "accel_x_mps2",
        "accel_y_mps2",
        "accel_z_mps2",

        # Gyroscope, rad/s
        "gyro_x_rps",
        "gyro_y_rps",
        "gyro_z_rps",

        # Orientation quaternion
        "quat_i",
        "quat_j",
        "quat_k",
        "quat_real",

        # Gravity vector, m/s^2
        "gravity_x_mps2",
        "gravity_y_mps2",
        "gravity_z_mps2",

        # Linear acceleration, m/s^2
        # useful backup, but raw accel is still the truth
        "lin_accel_x_mps2",
        "lin_accel_y_mps2",
        "lin_accel_z_mps2",
    ]

    start_ns = time.time_ns()
    sample_count = 0

    with imu_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while running:
            now_ns = time.time_ns()
            time_s = (now_ns - start_ns) / 1e9

            try:
                ax, ay, az = bno.acceleration
                gx, gy, gz = bno.gyro
                qi, qj, qk, qr = bno.quaternion
                grav_x, grav_y, grav_z = bno.gravity
                lax, lay, laz = bno.linear_acceleration

                writer.writerow({
                    "time_ns": now_ns,
                    "time_s": time_s,

                    "accel_x_mps2": ax,
                    "accel_y_mps2": ay,
                    "accel_z_mps2": az,

                    "gyro_x_rps": gx,
                    "gyro_y_rps": gy,
                    "gyro_z_rps": gz,

                    "quat_i": qi,
                    "quat_j": qj,
                    "quat_k": qk,
                    "quat_real": qr,

                    "gravity_x_mps2": grav_x,
                    "gravity_y_mps2": grav_y,
                    "gravity_z_mps2": grav_z,

                    "lin_accel_x_mps2": lax,
                    "lin_accel_y_mps2": lay,
                    "lin_accel_z_mps2": laz,
                })

                sample_count += 1

                # flush often enough so data survives if something fails
                if sample_count % 50 == 0:
                    f.flush()

            except Exception as e:
                print(f"IMU read warning: {e}")

            time.sleep(SAMPLE_DELAY)

        f.flush()

    print("IMU logging stopped cleanly")


if __name__ == "__main__":
    main()
