from pathlib import Path
import sys

if len(sys.argv) < 2:
    print("Usage: python3 prepare_slam_session.py ../data/sessions/session_001")
    raise SystemExit(1)

session_dir = Path(sys.argv[1])

if not session_dir.exists():
    print(f"Session does not exist: {session_dir}")
    raise SystemExit(1)

raw_dir = session_dir / "raw"
processed_dir = session_dir / "processed"
map_dir = session_dir / "map"
calibration_dir = session_dir / "calibration"
notes_dir = session_dir / "notes"

processed_dir.mkdir(exist_ok=True)
map_dir.mkdir(exist_ok=True)
calibration_dir.mkdir(exist_ok=True)
notes_dir.mkdir(exist_ok=True)

# placeholder files
(map_dir / "trajectory_visual.csv").touch()
(map_dir / "trajectory_visual_inertial.csv").touch()
(map_dir / "track_map.json").touch()
(calibration_dir / "camera_calibration.json").touch()
(notes_dir / "slam_notes.txt").touch()

print(f"SLAM-ready session prepared: {session_dir}")
