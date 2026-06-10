import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

# --- PATHS ---
BASE = Path.home() / "Developer" / "kart_project"
SESSIONS = BASE / "data" / "sessions"
SRC = BASE / "src"

SESSIONS.mkdir(parents=True, exist_ok=True)

# --- CREATE SESSION ---
existing = sorted(
    p for p in SESSIONS.iterdir()
    if p.is_dir() and p.name.startswith("session_")
)

next_num = len(existing) + 1
session_name = f"session_{next_num:03d}"
session_path = SESSIONS / session_name

raw = session_path / "raw"
processed = session_path / "processed"
notes = session_path / "notes"

raw.mkdir(parents=True, exist_ok=True)
processed.mkdir(parents=True, exist_ok=True)
notes.mkdir(parents=True, exist_ok=True)

video_file = raw / f"{session_name}_video.mp4"
imu_file = raw / f"{session_name}_imu.csv"
meta_file = session_path / "meta.json"

# --- START TIME ---
session_start_time_ns = time.time_ns()
session_start_time_unix = time.time()

# --- META ---
meta = {
    "session_id": session_name,
    "created_at": datetime.now().isoformat(),

    "session_start_time_ns": session_start_time_ns,
    "session_start_time_unix": session_start_time_unix,

    "video_file": f"raw/{session_name}_video.mp4",
    "imu_file": f"raw/{session_name}_imu.csv",

    "fps": 30,
    "resolution": "640x480",
    "camera_model": "Pi Camera 3 Wide",
    "imu_model": "BNO085",

    "imu_axis_convention": {
        "kart_frame": {
            "x": "forward",
            "y": "right",
            "z": "up"
        },
        "imu_mount_status": "temporary_not_final",
        "notes": "Raw IMU data is recorded in the sensor frame. Final sensor-to-kart transform will be added after permanent mounting."
    },

    "notes": ""
}

meta_file.write_text(json.dumps(meta, indent=2))

# --- START IMU RECORDING ---
imu_cmd = [
    "python3",
    str(SRC / "record_imu.py"),
    str(session_path)
]

imu_proc = subprocess.Popen(imu_cmd)

# small delay so IMU starts before video
time.sleep(0.5)

# --- START VIDEO RECORDING ---
video_start_time_ns = time.time_ns()

video_cmd = [
    "rpicam-vid",
    "-t", "600000",  # 10 minutes
    "--width", "640",
    "--height", "480",
    "--framerate", "30",
    "-o", str(video_file)
]

video_proc = subprocess.Popen(video_cmd)

# update meta with actual video start
meta["video_start_time_ns"] = video_start_time_ns
meta_file.write_text(json.dumps(meta, indent=2))

try:
    video_proc.wait()

except KeyboardInterrupt:
    print("Stopping session...")

finally:
    video_proc.terminate()
    imu_proc.terminate()

    video_proc.wait()
    imu_proc.wait()

    session_end_time_ns = time.time_ns()
    meta["session_end_time_ns"] = session_end_time_ns
    meta_file.write_text(json.dumps(meta, indent=2))

print(f"Saved session: {session_path}")
print(f"Video: {video_file}")
print(f"IMU: {imu_file}")
print(f"Meta: {meta_file}")
