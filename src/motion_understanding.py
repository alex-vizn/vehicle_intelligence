import sys
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path.home() / "Developer" / "kart_project"
SESSIONS_DIR = BASE_DIR / "data" / "sessions"


# ---------------- CONFIG ----------------

WINDOW_SECONDS = 0.5
MIN_EVENT_SECONDS = 0.8

ACCEL_ENTER_THRESHOLD = 0.45      # m/s^2, signal needed to enter accel/brake
ACCEL_EXIT_THRESHOLD = 0.25       # m/s^2, weaker signal allowed to stay in accel/brake

TURN_ACCEL_ENTER_THRESHOLD = 0.70 # m/s^2, stronger accel/brake needed while turning
TURN_ACCEL_EXIT_THRESHOLD = 0.40  # m/s^2

TURN_ENTER_THRESHOLD = 0.07       # rad/s
TURN_EXIT_THRESHOLD = 0.04        # rad/s

BUMP_THRESHOLD = 1.6              # m/s^2
VISUAL_MOTION_THRESHOLD = 0.8     # pixels/frame
VISUAL_SAMPLE_EVERY = 3

STANDING_CONFIRM_SECONDS = 1.0
GRAVITY = 9.81


# Sensor orientation from your setup:
# Y = forward/backward
# X = side force
# Z = vertical
FORWARD_COL = "lin_accel_y_mps2"
SIDE_COL = "lin_accel_x_mps2"
VERTICAL_COL = "lin_accel_z_mps2"
YAW_COL = "gyro_z_rps"

# Previous version was correct.
FORWARD_SIGN = 1.0
SIDE_SIGN = 1.0
YAW_SIGN = 1.0


# ---------------- SESSION LOADING ----------------

def get_session_path():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python3 src/motion_understanding.py session_020")

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


# ---------------- IMU PROCESSING ----------------

def validate_imu_columns(imu_df):
    required = [
        "session_time_s",
        FORWARD_COL,
        SIDE_COL,
        VERTICAL_COL,
        YAW_COL,
    ]

    missing = [c for c in required if c not in imu_df.columns]

    if missing:
        raise ValueError(f"Missing IMU columns: {missing}")


def estimate_sample_rate(imu_df):
    duration = imu_df["session_time_s"].max() - imu_df["session_time_s"].min()

    if duration <= 0:
        return 100.0

    return len(imu_df) / duration


def add_smoothed_signals(imu_df):
    imu_df = imu_df.copy()

    sample_rate = estimate_sample_rate(imu_df)
    window_rows = max(3, int(WINDOW_SECONDS * sample_rate))

    if window_rows % 2 == 0:
        window_rows += 1

    imu_df["forward_smooth"] = (
        imu_df[FORWARD_COL] * FORWARD_SIGN
    ).rolling(window_rows, center=True, min_periods=1).median()

    imu_df["side_smooth"] = (
        imu_df[SIDE_COL] * SIDE_SIGN
    ).rolling(window_rows, center=True, min_periods=1).median()

    imu_df["vertical_smooth"] = (
        imu_df[VERTICAL_COL]
    ).rolling(window_rows, center=True, min_periods=1).median()

    imu_df["yaw_smooth"] = (
        imu_df[YAW_COL] * YAW_SIGN
    ).rolling(window_rows, center=True, min_periods=1).median()

    imu_df["forward_g"] = imu_df["forward_smooth"] / GRAVITY
    imu_df["side_g"] = imu_df["side_smooth"] / GRAVITY
    imu_df["vertical_g"] = imu_df["vertical_smooth"] / GRAVITY

    dt = imu_df["session_time_s"].diff().replace(0, np.nan)

    imu_df["forward_jerk"] = imu_df["forward_smooth"].diff() / dt
    imu_df["side_jerk"] = imu_df["side_smooth"].diff() / dt
    imu_df["yaw_change_rate"] = imu_df["yaw_smooth"].diff() / dt

    imu_df["forward_jerk"] = imu_df["forward_jerk"].fillna(0)
    imu_df["side_jerk"] = imu_df["side_jerk"].fillna(0)
    imu_df["yaw_change_rate"] = imu_df["yaw_change_rate"].fillna(0)

    imu_df["smoothness_score"] = (
        imu_df["forward_jerk"].abs()
        + imu_df["side_jerk"].abs()
        + imu_df["yaw_change_rate"].abs()
    )

    imu_df["sample_rate_hz"] = sample_rate
    imu_df["smoothing_window_rows"] = window_rows

    return imu_df, sample_rate, window_rows


# ---------------- HYSTERESIS STATE ----------------

def update_hysteresis_state(row, previous_state):
    forward = row["forward_smooth"]
    yaw = row["yaw_smooth"]

    previous_longitudinal = previous_state.get("longitudinal", "NEUTRAL")
    previous_turn = previous_state.get("turn", "STRAIGHT")

    # Turning hysteresis
    if previous_turn == "LEFT":
        if yaw > TURN_EXIT_THRESHOLD:
            turn = "LEFT"
        else:
            turn = "STRAIGHT"
    elif previous_turn == "RIGHT":
        if yaw < -TURN_EXIT_THRESHOLD:
            turn = "RIGHT"
        else:
            turn = "STRAIGHT"
    else:
        if yaw > TURN_ENTER_THRESHOLD:
            turn = "LEFT"
        elif yaw < -TURN_ENTER_THRESHOLD:
            turn = "RIGHT"
        else:
            turn = "STRAIGHT"

    is_turning = turn in ["LEFT", "RIGHT"]

    enter_threshold = TURN_ACCEL_ENTER_THRESHOLD if is_turning else ACCEL_ENTER_THRESHOLD
    exit_threshold = TURN_ACCEL_EXIT_THRESHOLD if is_turning else ACCEL_EXIT_THRESHOLD

    # Accel/brake hysteresis
    if previous_longitudinal == "ACCELERATING":
        if forward > exit_threshold:
            longitudinal = "ACCELERATING"
        else:
            longitudinal = "NEUTRAL"

    elif previous_longitudinal == "BRAKING":
        if forward < -exit_threshold:
            longitudinal = "BRAKING"
        else:
            longitudinal = "NEUTRAL"

    else:
        if forward > enter_threshold:
            longitudinal = "ACCELERATING"
        elif forward < -enter_threshold:
            longitudinal = "BRAKING"
        else:
            longitudinal = "NEUTRAL"

    return {
        "longitudinal": longitudinal,
        "turn": turn,
    }


def label_from_state(state, visual_moving, standing_confirmed):
    longitudinal = state["longitudinal"]
    turn = state["turn"]

    if turn == "LEFT":
        if longitudinal == "BRAKING":
            return "BRAKING_LEFT"
        if longitudinal == "ACCELERATING":
            return "ACCELERATING_LEFT"
        return "TURNING_LEFT"

    if turn == "RIGHT":
        if longitudinal == "BRAKING":
            return "BRAKING_RIGHT"
        if longitudinal == "ACCELERATING":
            return "ACCELERATING_RIGHT"
        return "TURNING_RIGHT"

    if longitudinal == "BRAKING":
        return "BRAKING"

    if longitudinal == "ACCELERATING":
        return "ACCELERATING"

    if standing_confirmed:
        return "STANDING"

    # No more CRUISING_UNCONFIRMED.
    # If we cannot prove standing, we assume cruising/stable movement.
    return "CRUISING"


# ---------------- VISUAL MOTION ----------------

def calculate_visual_motion(prev_gray, gray, mask=None):
    points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=300,
        qualityLevel=0.02,
        minDistance=10,
        blockSize=7,
        mask=mask
    )

    if points is None or len(points) < 20:
        return 0.0

    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01)
    )

    if next_points is None:
        return 0.0

    status = status.flatten()
    error = error.flatten()

    valid = (status == 1) & (error < 30)

    good_old = points[valid].reshape(-1, 2)
    good_new = next_points[valid].reshape(-1, 2)

    if len(good_old) < 20:
        return 0.0

    motion = good_new - good_old
    magnitudes = np.linalg.norm(motion, axis=1)

    return float(np.median(magnitudes))


def create_motion_mask(gray):
    h, w = gray.shape

    mask = np.zeros_like(gray)

    # Use the upper/middle scene and avoid hood/dashboard.
    cv2.rectangle(mask, (0, 0), (w, int(h * 0.75)), 255, -1)

    return mask


# ---------------- STANDING LOGIC ----------------

def update_standing_timer(low_motion_and_calm, current_time_s, calm_start_time_s):
    if low_motion_and_calm:
        if calm_start_time_s is None:
            calm_start_time_s = current_time_s

        calm_duration_s = current_time_s - calm_start_time_s
        standing_confirmed = calm_duration_s >= STANDING_CONFIRM_SECONDS

    else:
        calm_start_time_s = None
        calm_duration_s = 0.0
        standing_confirmed = False

    return calm_start_time_s, calm_duration_s, standing_confirmed


# ---------------- TIMELINE PROCESSING ----------------

def merge_short_events(frame_labels, fps):
    if not frame_labels:
        return frame_labels

    min_frames = int(MIN_EVENT_SECONDS * fps)

    labels = [item.copy() for item in frame_labels]

    start = 0

    while start < len(labels):
        current = labels[start]["label"]
        end = start + 1

        while end < len(labels) and labels[end]["label"] == current:
            end += 1

        segment_len = end - start

        if segment_len < min_frames:
            prev_label = labels[start - 1]["label"] if start > 0 else None
            next_label = labels[end]["label"] if end < len(labels) else None

            replacement = prev_label or next_label or current

            for i in range(start, end):
                labels[i]["label"] = replacement

        start = end

    return labels


def build_event_timeline(frame_labels):
    if not frame_labels:
        return []

    events = []

    start_idx = 0
    current_label = frame_labels[0]["label"]
    current_bump = frame_labels[0]["bump"]

    for i in range(1, len(frame_labels)):
        label = frame_labels[i]["label"]
        bump = frame_labels[i]["bump"]

        if label != current_label or bump != current_bump:
            start_time = frame_labels[start_idx]["time_s"]
            end_time = frame_labels[i - 1]["time_s"]

            events.append({
                "start_s": round(start_time),
                "end_s": round(end_time),
                "duration_s": round(end_time - start_time, 2),
                "label": current_label,
                "bump": bool(current_bump),
            })

            start_idx = i
            current_label = label
            current_bump = bump

    start_time = frame_labels[start_idx]["time_s"]
    end_time = frame_labels[-1]["time_s"]

    events.append({
        "start_s": round(start_time),
        "end_s": round(end_time),
        "duration_s": round(end_time - start_time, 2),
        "label": current_label,
        "bump": bool(current_bump),
    })

    return events


def closest_imu_row(imu_df, frame_time_s):
    idx = (imu_df["session_time_s"] - frame_time_s).abs().idxmin()
    return imu_df.loc[idx]


# ---------------- PLOTS ----------------

def make_motion_plots(imu_df, output_path):
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    t = imu_df["session_time_s"]

    axes[0].plot(t, imu_df["forward_smooth"])
    axes[0].axhline(ACCEL_ENTER_THRESHOLD, linestyle="--", label="enter")
    axes[0].axhline(-ACCEL_ENTER_THRESHOLD, linestyle="--")
    axes[0].axhline(ACCEL_EXIT_THRESHOLD, linestyle=":", label="exit")
    axes[0].axhline(-ACCEL_EXIT_THRESHOLD, linestyle=":")
    axes[0].set_ylabel("forward/brake")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, imu_df["side_smooth"])
    axes[1].set_ylabel("side force")
    axes[1].grid(True)

    axes[2].plot(t, imu_df["yaw_smooth"])
    axes[2].axhline(TURN_ENTER_THRESHOLD, linestyle="--", label="enter")
    axes[2].axhline(-TURN_ENTER_THRESHOLD, linestyle="--")
    axes[2].axhline(TURN_EXIT_THRESHOLD, linestyle=":", label="exit")
    axes[2].axhline(-TURN_EXIT_THRESHOLD, linestyle=":")
    axes[2].set_ylabel("yaw/turn")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(t, imu_df["vertical_smooth"])
    axes[3].axhline(BUMP_THRESHOLD, linestyle="--")
    axes[3].axhline(-BUMP_THRESHOLD, linestyle="--")
    axes[3].set_ylabel("vertical/bump")
    axes[3].grid(True)

    axes[4].plot(t, imu_df["smoothness_score"])
    axes[4].set_ylabel("roughness")
    axes[4].set_xlabel("session time (s)")
    axes[4].grid(True)

    fig.suptitle("Motion Understanding Signals")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


# ---------------- MAIN ----------------

def main():
    session_path = get_session_path()
    meta = load_meta(session_path)
    video_path, imu_path = get_paths(session_path, meta)

    processed_dir = session_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    debug_video_path = processed_dir / f"{session_path.name}_motion_debug.mp4"
    timeline_path = processed_dir / f"{session_path.name}_motion_timeline.csv"
    summary_path = processed_dir / f"{session_path.name}_motion_summary.json"
    signals_plot_path = processed_dir / f"{session_path.name}_motion_signals.png"

    imu_df = pd.read_csv(imu_path)
    validate_imu_columns(imu_df)

    imu_df, sample_rate, window_rows = add_smoothed_signals(imu_df)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = float(meta.get("fps", 30))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_start_session_s = float(meta.get("video_start_session_s", 0.0))

    writer = cv2.VideoWriter(
        str(debug_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    ret, first_frame = cap.read()

    if not ret:
        raise RuntimeError("Could not read first video frame")

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    motion_mask = create_motion_mask(prev_gray)

    frame_labels = []
    last_visual_motion = 0.0
    frame_idx = 0

    calm_start_time_s = None
    calm_duration_s = 0.0

    motion_state = {
        "longitudinal": "NEUTRAL",
        "turn": "STRAIGHT",
    }

    while True:
        if frame_idx == 0:
            frame = first_frame
            ret = True
        else:
            ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_idx % VISUAL_SAMPLE_EVERY == 0 and frame_idx > 0:
            last_visual_motion = calculate_visual_motion(prev_gray, gray, mask=motion_mask)

        visual_moving = last_visual_motion > VISUAL_MOTION_THRESHOLD

        frame_time_s = video_start_session_s + (frame_idx / fps)
        imu_row = closest_imu_row(imu_df, frame_time_s)

        forward = float(imu_row["forward_smooth"])
        side = float(imu_row["side_smooth"])
        yaw = float(imu_row["yaw_smooth"])
        vertical = float(imu_row["vertical_smooth"])

        forward_g = float(imu_row["forward_g"])
        side_g = float(imu_row["side_g"])
        vertical_g = float(imu_row["vertical_g"])
        smoothness_score = float(imu_row["smoothness_score"])

        imu_calm = (
            abs(forward) < ACCEL_EXIT_THRESHOLD
            and abs(yaw) < TURN_EXIT_THRESHOLD
            and abs(vertical) < BUMP_THRESHOLD
        )

        low_motion_and_calm = (not visual_moving) and imu_calm

        calm_start_time_s, calm_duration_s, standing_confirmed = update_standing_timer(
            low_motion_and_calm,
            frame_time_s,
            calm_start_time_s
        )

        motion_state = update_hysteresis_state(imu_row, motion_state)
        label = label_from_state(motion_state, visual_moving, standing_confirmed)

        bump = abs(vertical) > BUMP_THRESHOLD

        frame_labels.append({
            "frame_idx": frame_idx,
            "time_s": frame_time_s,
            "label": label,
            "bump": bump,
            "visual_motion": last_visual_motion,
            "visual_moving": visual_moving,
            "standing_confirmed": standing_confirmed,
            "calm_duration_s": calm_duration_s,
            "forward": forward,
            "side": side,
            "yaw": yaw,
            "vertical": vertical,
            "forward_g": forward_g,
            "side_g": side_g,
            "vertical_g": vertical_g,
            "smoothness_score": smoothness_score,
        })

        overlay = frame.copy()

        display_label = label
        if bump:
            display_label += " + BUMP"

        lines = [
            f"Session: {session_path.name}",
            f"Frame: {frame_idx}/{frame_count}",
            f"Time: {frame_time_s:.2f}s",
            f"MOTION: {display_label}",
            f"visual motion: {last_visual_motion:.2f} px/frame | moving: {visual_moving}",
            f"standing: {standing_confirmed} ({calm_duration_s:.1f}s calm)",
            f"state: {motion_state['longitudinal']} + {motion_state['turn']}",
            f"forward/brake: {forward:.3f} m/s^2 ({forward_g:.3f}g)",
            f"side force: {side:.3f} m/s^2 ({side_g:.3f}g)",
            f"yaw: {yaw:.3f} rad/s",
            f"vertical: {vertical:.3f} m/s^2 ({vertical_g:.3f}g)",
            f"smoothness/roughness: {smoothness_score:.3f}",
        ]

        y = 35

        for text in lines:
            cv2.putText(
                overlay,
                text,
                (25, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (255, 255, 255),
                2
            )
            y += 26

        writer.write(overlay)

        prev_gray = gray
        frame_idx += 1

    cap.release()
    writer.release()

    smoothed_labels = merge_short_events(frame_labels, fps)
    events = build_event_timeline(smoothed_labels)

    timeline_df = pd.DataFrame(events)
    timeline_df.to_csv(timeline_path, index=False)

    make_motion_plots(imu_df, signals_plot_path)

    label_counts = {}
    for item in smoothed_labels:
        label_counts[item["label"]] = label_counts.get(item["label"], 0) + 1

    summary = {
        "session": session_path.name,
        "video_path": str(video_path),
        "imu_path": str(imu_path),

        "outputs": {
            "motion_debug_video": str(debug_video_path),
            "motion_timeline_csv": str(timeline_path),
            "motion_summary_json": str(summary_path),
            "motion_signals_plot": str(signals_plot_path),
        },

        "config": {
            "window_seconds": WINDOW_SECONDS,
            "window_rows": window_rows,
            "minimum_event_seconds": MIN_EVENT_SECONDS,

            "accel_enter_threshold_mps2": ACCEL_ENTER_THRESHOLD,
            "accel_exit_threshold_mps2": ACCEL_EXIT_THRESHOLD,
            "turn_accel_enter_threshold_mps2": TURN_ACCEL_ENTER_THRESHOLD,
            "turn_accel_exit_threshold_mps2": TURN_ACCEL_EXIT_THRESHOLD,

            "turn_enter_threshold_rps": TURN_ENTER_THRESHOLD,
            "turn_exit_threshold_rps": TURN_EXIT_THRESHOLD,

            "bump_threshold_mps2": BUMP_THRESHOLD,
            "visual_motion_threshold_px": VISUAL_MOTION_THRESHOLD,
            "standing_confirm_seconds": STANDING_CONFIRM_SECONDS,

            "forward_sign": FORWARD_SIGN,
            "side_sign": SIDE_SIGN,
            "yaw_sign": YAW_SIGN,
        },

        "sensor_interpretation": {
            "forward_axis": "Y linear acceleration",
            "side_axis": "X linear acceleration",
            "vertical_axis": "Z linear acceleration",
            "turn_axis": "Z gyroscope yaw",
            "note": "IMU detects acceleration, not true speed. Visual motion helps separate standing from cruising."
        },

        "metrics": {
            "forward_g_note": "longitudinal acceleration divided by 9.81",
            "side_g_note": "lateral force divided by 9.81",
            "smoothness_score_note": "rough estimate based on acceleration/yaw changes; higher means less smooth"
        },

        "video": {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
        },

        "imu": {
            "sample_rate_hz": sample_rate,
            "sample_count": int(len(imu_df)),
        },

        "label_frame_counts": label_counts,
        "events_count": len(events),
        "status": "ok"
    }

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Motion debug video: {debug_video_path}")
    print(f"Motion timeline CSV: {timeline_path}")
    print(f"Motion summary JSON: {summary_path}")
    print(f"Motion signals plot: {signals_plot_path}")


if __name__ == "__main__":
    main()
