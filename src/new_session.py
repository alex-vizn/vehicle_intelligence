from pathlib import Path
import json
from datetime import datetime

base = Path("data/sessions")
base.mkdir(parents=True, exist_ok=True)

existing = sorted(
    p for p in base.iterdir()
    if p.is_dir() and p.name.startswith("session_")
)

next_num = len(existing) + 1
session_name = f"session_{next_num:03d}"
session_path = base / session_name

raw = session_path / "raw"
processed = session_path / "processed"
notes = session_path / "notes"

raw.mkdir(parents=True, exist_ok=True)
processed.mkdir(parents=True, exist_ok=True)
notes.mkdir(parents=True, exist_ok=True)

video_file_name = f"video_{next_num:03d}.mp4"
imu_file_name = f"imu_{next_num:03d}.csv"

imu_file = raw / imu_file_name
meta_file = session_path / "meta.json"

# create empty IMU file with header
imu_file.write_text("time_ns,ax,ay,az,gx,gy,gz,lax,lay,laz,mx,my,mz,qw,qx,qy,qz\n")

# create meta.json
meta = {
    "session_id": session_name,
    "created_at": datetime.now().isoformat(),
    "video_file": f"raw/{video_file_name}",
    "imu_file": f"raw/{imu_file_name}",
    "fps": 30,
    "resolution": "640x480",
    "camera_model": "Pi Camera 3 Wide",
    "imu_model": "BNO085",
    "coordinate_system": {
        "x": "front",
        "y": "right",
        "z": "up"
    },
    "notes": ""
}

meta_file.write_text(json.dumps(meta, indent=2))

print(session_path)