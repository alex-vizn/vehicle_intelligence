import cv2
import numpy as np
from pathlib import Path
from collections import deque, Counter

video_path = Path("../data/sessions/session_001/raw/pov_kart.mov")

session_dir = video_path.parent.parent
session_name = session_dir.name
processed_dir = session_dir / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

output_path = processed_dir / f"{session_name}_vision.mp4"

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

# temporal filtering
alpha = 0.6
prev_center_x1 = None
prev_center_x2 = None
prev_center_line = None
direction_history = deque(maxlen=3)

# visual tracking memory
prev_gray = None
prev_points = None

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    overlay = frame.copy()

    # 1. grayscale + blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    h, w = blur.shape
    roi_center_x = w // 2

    # 2. two-zone ROI mask
    y_start = int(h * 0.30)
    y_end = int(h * 0.80)

    mask = np.zeros_like(gray)

    left_zone = np.array([[
        (int(w * 0.00), y_end),
        (int(w * 0.00), y_start),
        (int(w * 0.40), y_start),
        (int(w * 0.28), y_end)
    ]], dtype=np.int32)

    right_zone = np.array([[
        (int(w * 0.72), y_end),
        (int(w * 0.60), y_start),
        (int(w * 1.00), y_start),
        (int(w * 1.00), y_end)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, left_zone, 255)
    cv2.fillPoly(mask, right_zone, 255)

    # draw ROI zones
    cv2.polylines(overlay, left_zone, True, (180, 180, 180), 1)
    cv2.polylines(overlay, right_zone, True, (180, 180, 180), 1)
    cv2.line(overlay, (roi_center_x, y_start), (roi_center_x, y_end), (180, 180, 180), 1)

    # 3. color mask: white + red
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 160]),
        np.array([180, 70, 255])
    )

    red_mask1 = cv2.inRange(
        hsv,
        np.array([0, 80, 80]),
        np.array([10, 255, 255])
    )

    red_mask2 = cv2.inRange(
        hsv,
        np.array([170, 80, 80]),
        np.array([180, 255, 255])
    )

    black_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 0]),
        np.array([180, 255, 95])
    )

    color_mask = cv2.bitwise_or(white_mask, red_mask1)
    color_mask = cv2.bitwise_or(color_mask, red_mask2)
    color_mask = cv2.bitwise_or(color_mask, black_mask)
    color_mask = cv2.bitwise_or(color_mask, white_mask)
    color_mask = cv2.bitwise_and(color_mask, mask)

    kernel = np.ones((5, 5), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

    # 4. edges: normal edges + color edges
    gray_edges = cv2.Canny(blur, 75, 200)
    gray_edges = cv2.bitwise_and(gray_edges, mask)

    color_edges = cv2.Canny(color_mask, 50, 150)

    edges = cv2.bitwise_or(gray_edges, color_edges)

    # 5. visual tracking: estimate frame-to-frame motion
    flow_dx = 0
    flow_dy = 0
    flow_valid = False

    if prev_gray is not None and prev_points is not None and len(prev_points) > 0:
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            prev_points,
            None
        )

        if next_points is not None:
            good_new = next_points[status.flatten() == 1].reshape(-1, 2)
            good_old = prev_points[status.flatten() == 1].reshape(-1, 2)

            if len(good_new) > 5:
                motion = good_new - good_old
                flow_dx = int(np.median(motion[:, 0]))
                flow_dy = int(np.median(motion[:, 1]))
                flow_valid = True

    # 6. Hough lines
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=40,
        minLineLength=20,
        maxLineGap=40
    )

    left_lines = []
    right_lines = []
    line_view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            dx = x2 - x1
            dy = y2 - y1
            length = (dx**2 + dy**2) ** 0.5

            if dx == 0:
                continue

            angle_deg = abs(np.degrees(np.arctan2(dy, dx)))

            if length < 40:
                continue

            if angle_deg < 20 or angle_deg > 80:
                continue

            mid_x = (x1 + x2) / 2

            if mid_x < roi_center_x:
                left_lines.append((x1, y1, x2, y2))
                cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.line(line_view, (x1, y1), (x2, y2), (255, 0, 0), 2)
            else:
                right_lines.append((x1, y1, x2, y2))
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.line(line_view, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 7. average boundaries
    left_avg = None
    right_avg = None

    if left_lines:
        left_avg = (
            int(np.mean([l[0] for l in left_lines])),
            int(np.mean([l[1] for l in left_lines])),
            int(np.mean([l[2] for l in left_lines])),
            int(np.mean([l[3] for l in left_lines]))
        )

    if right_lines:
        right_avg = (
            int(np.mean([l[0] for l in right_lines])),
            int(np.mean([l[1] for l in right_lines])),
            int(np.mean([l[2] for l in right_lines])),
            int(np.mean([l[3] for l in right_lines]))
        )

    direction_raw = "UNKNOWN"
    direction_smooth = "UNKNOWN"
    offset = None
    confidence = "WEAK"

    center_detected = False
    # 8. compute center from detected boundaries
    if left_avg is not None and right_avg is not None:
        cx1 = int((left_avg[0] + right_avg[0]) / 2)
        cy1 = int((left_avg[1] + right_avg[1]) / 2)
        cx2 = int((left_avg[2] + right_avg[2]) / 2)
        cy2 = int((left_avg[3] + right_avg[3]) / 2)

        center_x1 = int(0.75 * cx1 + 0.25 * cx2)
        center_x2 = int(0.25 * cx1 + 0.75 * cx2)

        if prev_center_x1 is None:
            smooth_x1 = center_x1
            smooth_x2 = center_x2
        else:
            smooth_x1 = int(alpha * prev_center_x1 + (1 - alpha) * center_x1)
            smooth_x2 = int(alpha * prev_center_x2 + (1 - alpha) * center_x2)

        prev_center_x1 = smooth_x1
        prev_center_x2 = smooth_x2
        prev_center_line = (smooth_x1, cy1, smooth_x2, cy2)

        center_detected = True
        confidence = "GOOD" if len(left_lines) >= 2 and len(right_lines) >= 2 else "WEAK"

    # 9. fallback: use visual tracking if detection fails
    elif prev_center_line is not None and flow_valid:
        x1, y1, x2, y2 = prev_center_line
        smooth_x1 = x1 + flow_dx
        smooth_x2 = x2 + flow_dx
        cy1 = y1 + flow_dy
        cy2 = y2 + flow_dy
        prev_center_line = (smooth_x1, cy1, smooth_x2, cy2)
        confidence = "TRACKED"

    else:
        smooth_x1 = smooth_x2 = cy1 = cy2 = None

    # 10. direction + drawing
    if smooth_x1 is not None:
        cv2.line(
            overlay,
            (smooth_x1, cy1),
            (smooth_x2, cy2),
            (0, 255, 0),
            4
        )

        cv2.line(
            line_view,
            (smooth_x1, cy1),
            (smooth_x2, cy2),
            (0, 255, 0),
            4
        )

        center_mid_x = (smooth_x1 + smooth_x2) / 2
        offset = int(center_mid_x - roi_center_x)

        if offset < -20:
            direction_raw = "LEFT"
        elif offset > 20:
            direction_raw = "RIGHT"
        else:
            direction_raw = "STRAIGHT"

        direction_history.append(direction_raw)
    else:
        direction_history.append("UNKNOWN")

    direction_smooth = Counter(direction_history).most_common(1)[0][0]

    # 11. text overlay
    cv2.putText(overlay, direction_smooth, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(overlay, f"frame: {frame_idx}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, f"L:{len(left_lines)} R:{len(right_lines)}", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, f"conf: {confidence}", (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    offset_text = "offset: N/A" if offset is None else f"offset: {offset}px"
    cv2.putText(overlay, offset_text, (30, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # show windows
    cv2.imshow("mask", mask)
    cv2.imshow("color_mask", color_mask)
    cv2.imshow("edges", edges)
    cv2.imshow("vision_overlay", overlay)

    writer.write(overlay)

    # update visual tracking points for next frame
    prev_gray = gray.copy()
    prev_points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=7,
        mask=mask
    )

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"Saved output video to: {output_path}")