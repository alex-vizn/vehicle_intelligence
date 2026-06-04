import sys
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path.home() / "Developer" / "kart_project"
SESSIONS_DIR = BASE_DIR / "data" / "sessions"


def get_session_path():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python3 src/analyze_session.py session_020")

    session_name = sys.argv[1]
    session_path = SESSIONS_DIR / session_name

    if not session_path.exists():
        raise FileNotFoundError(f"Session not found: {session_path}")

    return session_path


def load_meta(session_path):
    meta_path = session_path / "meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")

    with meta_path.open("r") as f:
        return json.load(f)


def get_paths(session_path, meta):
    video_path = session_path / meta["video_file"]
    imu_path = session_path / meta["imu_file"]

    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")

    if not imu_path.exists():
        raise FileNotFoundError(f"Missing IMU CSV: {imu_path}")

    return video_path, imu_path


def closest_imu_row(imu_df, frame_time_s):
    idx = (imu_df["session_time_s"] - frame_time_s).abs().idxmin()
    return imu_df.loc[idx]


def motion_clue(row):
    forward_accel = row["lin_accel_y_mps2"]
    side_accel = row["lin_accel_x_mps2"]
    vertical_accel = row["lin_accel_z_mps2"]
    yaw_rate = row["gyro_z_rps"]

    clues = []

    if forward_accel > 0.8:
        clues.append("FORWARD +")
    elif forward_accel < -0.8:
        clues.append("FORWARD -")

    if side_accel > 0.8:
        clues.append("RIGHT FORCE")
    elif side_accel < -0.8:
        clues.append("LEFT FORCE")

    if yaw_rate > 0.08:
        clues.append("TURN +")
    elif yaw_rate < -0.08:
        clues.append("TURN -")

    if abs(vertical_accel) > 1.2:
        clues.append("BUMP")

    if not clues:
        return "stable"

    return " | ".join(clues)


def make_imu_plots(imu_df, output_path):
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    t = imu_df["session_time_s"]

    axes[0].plot(t, imu_df["lin_accel_y_mps2"])
    axes[0].set_ylabel("Forward accel Y")
    axes[0].grid(True)

    axes[1].plot(t, imu_df["lin_accel_x_mps2"])
    axes[1].set_ylabel("Side accel X")
    axes[1].grid(True)

    axes[2].plot(t, imu_df["lin_accel_z_mps2"])
    axes[2].set_ylabel("Vertical accel Z")
    axes[2].grid(True)

    axes[3].plot(t, imu_df["gyro_z_rps"])
    axes[3].set_ylabel("Yaw gyro Z")
    axes[3].set_xlabel("Session time (s)")
    axes[3].grid(True)

    fig.suptitle("IMU Motion Signals")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    session_path = get_session_path()
    meta = load_meta(session_path)
    video_path, imu_path = get_paths(session_path, meta)

    processed_dir = session_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    debug_video_path = processed_dir / f"{session_path.name}_debug_sync.mp4"
    plot_path = processed_dir / f"{session_path.name}_imu_plots.png"
    report_path = processed_dir / f"{session_path.name}_validation_report.json"

    imu_df = pd.read_csv(imu_path)

    required_columns = [
        "session_time_s",
        "lin_accel_x_mps2",
        "lin_accel_y_mps2",
        "lin_accel_z_mps2",
        "gyro_x_rps",
        "gyro_y_rps",
        "gyro_z_rps",
        "gravity_x_mps2",
        "gravity_y_mps2",
        "gravity_z_mps2",
    ]

    missing = [c for c in required_columns if c not in imu_df.columns]
    if missing:
        raise ValueError(f"Missing IMU columns: {missing}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = float(meta.get("fps", 30))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_duration_s = frame_count / fps if fps > 0 else 0
    imu_duration_s = float(imu_df["session_time_s"].max() - imu_df["session_time_s"].min())
    imu_sample_rate = len(imu_df) / imu_duration_s if imu_duration_s > 0 else 0

    video_start_session_s = float(meta.get("video_start_session_s", 0.0))

    make_imu_plots(imu_df, plot_path)

    writer = cv2.VideoWriter(
        str(debug_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    sample_every = 1
    frame_idx = 0
    sync_errors = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_time_s = video_start_session_s + (frame_idx / fps)
        row = closest_imu_row(imu_df, frame_time_s)

        sync_error = abs(float(row["session_time_s"]) - frame_time_s)
        sync_errors.append(sync_error)

        clue = motion_clue(row)

        overlay = frame.copy()

        lines = [
            f"Session: {session_path.name}",
            f"Frame: {frame_idx} / {frame_count}",
            f"Video time: {frame_time_s:.3f}s",
            f"IMU time: {row['session_time_s']:.3f}s",
            f"Sync error: {sync_error*1000:.1f} ms",
            f"Forward accel Y: {row['lin_accel_y_mps2']:.3f} m/s^2",
            f"Side accel X: {row['lin_accel_x_mps2']:.3f} m/s^2",
            f"Vertical accel Z: {row['lin_accel_z_mps2']:.3f} m/s^2",
            f"Gyro Z yaw: {row['gyro_z_rps']:.3f} rad/s",
            f"Motion: {clue}",
        ]

        y = 35
        for text in lines:
            cv2.putText(
                overlay,
                text,
                (25, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            y += 28

        writer.write(overlay)
        frame_idx += 1

    cap.release()
    writer.release()

    report = {
        "session": session_path.name,
        "video_path": str(video_path),
        "imu_path": str(imu_path),
        "debug_video": str(debug_video_path),
        "imu_plot": str(plot_path),

        "video_fps": fps,
        "video_frame_count": frame_count,
        "video_width": width,
        "video_height": height,
        "video_duration_s": video_duration_s,

        "imu_sample_count": int(len(imu_df)),
        "imu_duration_s": imu_duration_s,
        "imu_sample_rate_hz": imu_sample_rate,

        "video_start_session_s": video_start_session_s,
        "mean_sync_error_ms": float(np.mean(sync_errors) * 1000) if sync_errors else None,
        "max_sync_error_ms": float(np.max(sync_errors) * 1000) if sync_errors else None,

        "sensor_orientation_assumption": {
            "y": "forward/backward",
            "x": "right/left",
            "z": "vertical",
            "gyro_z": "yaw/turning"
        },

        "status": "ok"
    }

    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    print("Done.")
    print(f"Debug video: {debug_video_path}")
    print(f"IMU plots: {plot_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
