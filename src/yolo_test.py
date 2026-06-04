from ultralytics import YOLO
import cv2
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python3 src/yolo_test.py session_020")
    sys.exit(1)

session_name = sys.argv[1]

base = Path.home() / "Developer" / "kart_project"
session_dir = base / "data" / "sessions" / session_name

video_path = session_dir / "raw" / f"{session_name}_video.mp4"

output_path = (
    session_dir
    / "processed"
    / f"{session_name}_yolo_debug.mp4"
)

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(str(video_path))

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)

    annotated = results[0].plot()

    writer.write(annotated)

    frame_count += 1

    if frame_count % 100 == 0:
        print(frame_count)

cap.release()
writer.release()

print()
print("Done")
print(output_path)
