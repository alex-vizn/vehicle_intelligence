import cv2
import numpy as np
from pathlib import Path

video_path = Path("../data/sessions/session_001/raw/pov_kart_k1speed.mov")

session_dir = video_path.parent.parent
processed_dir = session_dir / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

output_path = processed_dir / "feature_tracking.mp4"

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print(f"Could not open video: {video_path}")
    raise SystemExit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_w, frame_h)
)

ret, first_frame = cap.read()
if not ret:
    print("Could not read first frame")
    raise SystemExit(1)

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

h, w = prev_gray.shape

feature_mask = np.zeros_like(prev_gray)

# allow upper / side environment
cv2.rectangle(
    feature_mask,
    (0, 0),
    (w, int(h * 0.65)),
    255,
    -1
)

# block lower center: kart / hands / steering wheel
kart_block = np.array([[
    (int(w * 0.20), h),
    (int(w * 0.35), int(h * 0.45)),
    (int(w * 0.65), int(h * 0.45)),
    (int(w * 0.80), h)
]], dtype=np.int32)

cv2.fillPoly(feature_mask, kart_block, 0)

# Find initial feature points
prev_points = cv2.goodFeaturesToTrack(
    prev_gray,
    maxCorners=250,
    qualityLevel=0.02,
    minDistance=12,
    blockSize=7,
    mask=feature_mask
)

# Lucas-Kanade optical flow settings
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

track_mask = np.zeros_like(first_frame)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output = frame.copy()

    if prev_points is not None and len(prev_points) > 0:
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            prev_points,
            None,
            **lk_params
        )

        if next_points is not None:
            good_new = next_points[status.flatten() == 1]
            good_old = prev_points[status.flatten() == 1]

            for new, old in zip(good_new, good_old):
                x_new, y_new = new.ravel()
                x_old, y_old = old.ravel()

                cv2.line(
                    track_mask,
                    (int(x_old), int(y_old)),
                    (int(x_new), int(y_new)),
                    (255, 0, 0),
                    2
                )

                cv2.circle(
                    output,
                    (int(x_new), int(y_new)),
                    4,
                    (0, 255, 0),
                    -1
                )

            output = cv2.add(output, track_mask)

            prev_points = good_new.reshape(-1, 1, 2)
            prev_gray = gray.copy()

    # Re-detect features if too few remain
    if prev_points is None or len(prev_points) < 80:
        prev_points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=250,
            qualityLevel=0.02,
            minDistance=12,
            blockSize=7,
            mask=feature_mask
        )
        track_mask = np.zeros_like(frame)
        prev_gray = gray.copy()

    cv2.putText(
        output,
        f"Frame: {frame_idx}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    count = 0 if prev_points is None else len(prev_points)
    cv2.putText(
        output,
        f"Tracked points: {count}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("feature_tracking", output)
    cv2.imshow("feature_mask", feature_mask)
    writer.write(output)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"Saved feature tracking video to: {output_path}")
