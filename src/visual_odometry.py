import cv2
import numpy as np
from pathlib import Path

video_path = Path("../data/sessions/session_001/raw/pov_kart_k1speed.mov")

session_dir = video_path.parent.parent
processed_dir = session_dir / "processed"
map_dir = session_dir / "map"

processed_dir.mkdir(parents=True, exist_ok=True)
map_dir.mkdir(parents=True, exist_ok=True)

output_video_path = processed_dir / "visual_odometry.mp4"
trajectory_csv_path = map_dir / "trajectory_visual.csv"

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
    str(output_video_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_w, frame_h)
)

ret, prev_frame = cap.read()
if not ret:
    print("Could not read first frame")
    raise SystemExit(1)

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

h, w = prev_gray.shape

feature_mask = np.zeros_like(prev_gray)

cv2.rectangle(
    feature_mask,
    (0, 0),
    (w, int(h * 0.70)),
    255,
    -1
)

kart_block = np.array([[
    (int(w * 0.15), h),
    (int(w * 0.35), int(h * 0.45)),
    (int(w * 0.65), int(h * 0.45)),
    (int(w * 0.85), h)
]], dtype=np.int32)

cv2.fillPoly(feature_mask, kart_block, 0)

prev_points = cv2.goodFeaturesToTrack(
    prev_gray,
    maxCorners=400,
    qualityLevel=0.02,
    minDistance=12,
    blockSize=7,
    mask=feature_mask
)

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# rough 2D trajectory state
x_pos = 0.0
y_pos = 0.0
heading = 0.0

trajectory = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    time_s = frame_idx / fps

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    overlay = frame.copy()

    dx_median = 0.0
    dy_median = 0.0
    rotation_est = 0.0
    tracked_count = 0

    if prev_points is not None and len(prev_points) > 10:
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            prev_points,
            None,
            **lk_params
        )

        if next_points is not None:
            status_flat = status.flatten()
            error_flat = error.flatten()

            valid = (status_flat == 1) & (error_flat < 20)

            good_new = next_points[valid].reshape(-1, 2)
            good_old = prev_points[valid].reshape(-1, 2)

            if len(good_new) >= 8:
                H, inlier_mask = cv2.findHomography(
                    good_new,
                    good_old,
                    cv2.RANSAC,
                    5.0
                )

                if inlier_mask is not None:
                    inliers = inlier_mask.ravel() == 1
                    good_new = good_new[inliers]
                    good_old = good_old[inliers]

            tracked_count = len(good_new)

            if tracked_count >= 12:
                motion = good_new - good_old

                dx_median = float(np.median(motion[:, 0]))
                dy_median = float(np.median(motion[:, 1]))

                # estimate rotation from left/right average motion difference
                left_motion = motion[good_old[:, 0] < w / 2]
                right_motion = motion[good_old[:, 0] >= w / 2]

                if len(left_motion) > 5 and len(right_motion) > 5:
                    left_dx = np.median(left_motion[:, 0])
                    right_dx = np.median(right_motion[:, 0])
                    rotation_est = float((right_dx - left_dx) / w)

                # rough visual odometry update
                heading += rotation_est

                forward_step = -dy_median * 0.01
                side_step = -dx_median * 0.005

                x_pos += forward_step * np.cos(heading) - side_step * np.sin(heading)
                y_pos += forward_step * np.sin(heading) + side_step * np.cos(heading)

                for new, old in zip(good_new, good_old):
                    x_new, y_new = new
                    x_old, y_old = old
                    cv2.circle(overlay, (int(x_new), int(y_new)), 3, (0, 255, 0), -1)
                    cv2.line(
                        overlay,
                        (int(x_old), int(y_old)),
                        (int(x_new), int(y_new)),
                        (255, 0, 0),
                        2
                    )

                prev_points = good_new.reshape(-1, 1, 2)

    if prev_points is None or len(prev_points) < 80:
        prev_points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=400,
            qualityLevel=0.02,
            minDistance=12,
            blockSize=7,
            mask=feature_mask
        )

    prev_gray = gray.copy()

    trajectory.append([frame_idx, time_s, x_pos, y_pos, heading, tracked_count])

    cv2.putText(
        overlay,
        f"frame: {frame_idx}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        overlay,
        f"tracked: {tracked_count}",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        overlay,
        f"x:{x_pos:.2f} y:{y_pos:.2f} heading:{heading:.2f}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        6
    )

    traj_view = np.zeros((250,250,3), dtype=np.uint8)

    scale = 20
    cx = 125
    cy = 125
    # mini top-down trajectory map
    traj_view = np.zeros((250, 250, 3), dtype=np.uint8)

    scale = 20
    cx = 125
    cy = 125

    for row in trajectory[-300:]:
        px = int(cx + row[2] * scale)
        py = int(cy + row[3] * scale)

        if 0 <= px < 250 and 0 <= py < 250:
            cv2.circle(traj_view, (px, py), 2, (0, 255, 0), -1)

    cv2.rectangle(traj_view, (0, 0), (249, 249), (180, 180, 180), 1)
    cv2.putText(traj_view, "Trajectory", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    overlay[20:270, frame_w - 270:frame_w - 20] = traj_view

    cv2.imshow("visual_odometry", overlay)
    writer.write(overlay)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

with open(trajectory_csv_path, "w") as f:
    f.write("frame_idx,time_s,x,y,heading,tracked_points\n")
    for row in trajectory:
        f.write(",".join(map(str, row)) + "\n")

print(f"Saved visual odometry video to: {output_video_path}")
print(f"Saved trajectory CSV to: {trajectory_csv_path}")