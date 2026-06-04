import sys
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


BASE_DIR = Path.home() / "Developer" / "kart_project"
SESSIONS_DIR = BASE_DIR / "data" / "sessions"


# ============================================================
# CONFIG
# ============================================================

FRAME_SAMPLE_EVERY = 1

# Region where road can realistically appear.
ROI_TOP_RATIO = 0.38
ROI_BOTTOM_RATIO = 0.98
ROI_LEFT_RATIO = 0.02
ROI_RIGHT_RATIO = 0.98

# Wider road sample patch.
# Before it was too narrow, so the system learned from a tiny patch.
SAMPLE_TOP_RATIO = 0.64
SAMPLE_BOTTOM_RATIO = 0.92
SAMPLE_LEFT_RATIO = 0.20
SAMPLE_RIGHT_RATIO = 0.80

# Color similarity tolerances.
HSV_H_TOL = 20
HSV_S_TOL = 65
HSV_V_TOL = 80

LAB_L_TOL = 80
LAB_A_TOL = 24
LAB_B_TOL = 28

# Mask thresholding.
LIKELIHOOD_PERCENTILE = 45
MIN_THRESHOLD = 0.32
MAX_THRESHOLD = 0.70

# Mask processing.
MIN_COMPONENT_AREA_RATIO = 0.010
MORPH_CLOSE_SIZE = 15
MORPH_OPEN_SIZE = 5

# Boundary extraction.
BOUNDARY_ROWS = 24
MIN_ROW_PIXELS = 35
MIN_ROAD_WIDTH_RATIO = 0.22
MAX_ROAD_WIDTH_RATIO = 0.98
MAX_ROW_FRAGMENT_COUNT = 5

# Bottom-road sanity check.
BOTTOM_CHECK_TOP_RATIO = 0.78
BOTTOM_CHECK_BOTTOM_RATIO = 0.96
MIN_BOTTOM_WIDTH_RATIO = 0.35
MIN_BOTTOM_AREA_RATIO = 0.12

# Curve fitting.
POLY_DEGREE = 2
MIN_POINTS_FOR_FIT = 6

# Temporal smoothing.
TEMPORAL_ALPHA = 0.70
MAX_CENTER_JUMP_RATIO = 0.18

# Confidence.
LOW_CONFIDENCE_THRESHOLD = 0.45
GOOD_CONFIDENCE_THRESHOLD = 0.70

# Drawing.
MASK_ALPHA = 0.42


# ============================================================
# SESSION LOADING
# ============================================================

def get_session_path():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python3 src/environment_perception.py session_020")

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


def get_video_path(session_path, meta):
    video_path = session_path / meta["video_file"]

    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")

    return video_path


# ============================================================
# ROI + ROAD SAMPLE
# ============================================================

def make_roi_mask(height, width):
    mask = np.zeros((height, width), dtype=np.uint8)

    top = int(height * ROI_TOP_RATIO)
    bottom = int(height * ROI_BOTTOM_RATIO)
    left = int(width * ROI_LEFT_RATIO)
    right = int(width * ROI_RIGHT_RATIO)

    polygon = np.array([[
        (left, bottom),
        (int(width * 0.18), top),
        (int(width * 0.82), top),
        (right, bottom),
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return mask


def get_sample_patch(frame):
    h, w = frame.shape[:2]

    y1 = int(h * SAMPLE_TOP_RATIO)
    y2 = int(h * SAMPLE_BOTTOM_RATIO)
    x1 = int(w * SAMPLE_LEFT_RATIO)
    x2 = int(w * SAMPLE_RIGHT_RATIO)

    return frame[y1:y2, x1:x2]


def sample_quality(sample):
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

    brightness_mean = float(np.mean(gray))
    brightness_std = float(np.std(gray))

    quality = 1.0

    if brightness_mean < 25 or brightness_mean > 235:
        quality *= 0.45

    if brightness_std > 60:
        quality *= 0.70

    return float(np.clip(quality, 0.0, 1.0))


# ============================================================
# ROAD MASK
# ============================================================

def adaptive_road_likelihood(frame, roi_mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    sample = get_sample_patch(frame)
    sample_hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    sample_lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)

    sample_q = sample_quality(sample)

    hsv_med = np.median(sample_hsv.reshape(-1, 3), axis=0)
    lab_med = np.median(sample_lab.reshape(-1, 3), axis=0)

    hsv_diff = np.abs(hsv.astype(np.float32) - hsv_med.astype(np.float32))
    lab_diff = np.abs(lab.astype(np.float32) - lab_med.astype(np.float32))

    h_diff = hsv_diff[:, :, 0]
    h_diff = np.minimum(h_diff, 180 - h_diff)

    hsv_score = (
        1.0
        - 0.35 * np.clip(h_diff / HSV_H_TOL, 0, 1)
        - 0.30 * np.clip(hsv_diff[:, :, 1] / HSV_S_TOL, 0, 1)
        - 0.35 * np.clip(hsv_diff[:, :, 2] / HSV_V_TOL, 0, 1)
    )

    lab_score = (
        1.0
        - 0.40 * np.clip(lab_diff[:, :, 0] / LAB_L_TOL, 0, 1)
        - 0.30 * np.clip(lab_diff[:, :, 1] / LAB_A_TOL, 0, 1)
        - 0.30 * np.clip(lab_diff[:, :, 2] / LAB_B_TOL, 0, 1)
    )

    likelihood = 0.5 * hsv_score + 0.5 * lab_score
    likelihood = np.clip(likelihood, 0.0, 1.0)

    likelihood[roi_mask == 0] = 0.0

    return likelihood, sample_q


def likelihood_to_mask(likelihood):
    nonzero = likelihood[likelihood > 0]

    if len(nonzero) == 0:
        return np.zeros_like(likelihood, dtype=np.uint8), 0.0

    threshold = np.percentile(nonzero, LIKELIHOOD_PERCENTILE)
    threshold = float(np.clip(threshold, MIN_THRESHOLD, MAX_THRESHOLD))

    mask = np.zeros_like(likelihood, dtype=np.uint8)
    mask[likelihood >= threshold] = 255

    return mask, threshold


def clean_mask(mask, width, height):
    close_kernel = np.ones((MORPH_CLOSE_SIZE, MORPH_CLOSE_SIZE), np.uint8)
    open_kernel = np.ones((MORPH_OPEN_SIZE, MORPH_OPEN_SIZE), np.uint8)

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    if num_labels <= 1:
        return cleaned, 0, 0.0

    min_area = int(width * height * MIN_COMPONENT_AREA_RATIO)

    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((label, area))

    if not components:
        return np.zeros_like(mask), num_labels - 1, 1.0

    # Keep the largest component, but confidence will punish if it is tiny/wrong.
    largest_label, largest_area = max(components, key=lambda x: x[1])

    largest = np.zeros_like(mask)
    largest[labels == largest_label] = 255

    fragmentation_score = min(1.0, (len(components) - 1) / 5.0)

    return largest, len(components), fragmentation_score


# ============================================================
# BOTTOM COVERAGE CHECK
# ============================================================

def bottom_road_metrics(mask):
    h, w = mask.shape

    y1 = int(h * BOTTOM_CHECK_TOP_RATIO)
    y2 = int(h * BOTTOM_CHECK_BOTTOM_RATIO)

    bottom_region = mask[y1:y2, :]

    area_ratio = float(np.count_nonzero(bottom_region)) / max(bottom_region.size, 1)

    widths = []

    for row in bottom_region:
        xs = np.where(row > 0)[0]
        if len(xs) > 0:
            widths.append(xs[-1] - xs[0])

    if widths:
        mean_width_ratio = float(np.mean(widths)) / w
        max_width_ratio = float(np.max(widths)) / w
    else:
        mean_width_ratio = 0.0
        max_width_ratio = 0.0

    return {
        "bottom_area_ratio": area_ratio,
        "bottom_mean_width_ratio": mean_width_ratio,
        "bottom_max_width_ratio": max_width_ratio,
    }


# ============================================================
# BOUNDARIES
# ============================================================

def count_row_fragments(row_binary):
    xs = np.where(row_binary > 0)[0]

    if len(xs) == 0:
        return 0

    gaps = np.diff(xs)
    return int(np.sum(gaps > 8) + 1)


def extract_boundaries(mask):
    h, w = mask.shape

    y_start = int(h * 0.44)
    y_end = int(h * 0.94)

    rows_y = np.linspace(y_start, y_end, BOUNDARY_ROWS).astype(int)

    left_points = []
    right_points = []
    center_points = []
    row_widths = []
    rejected_rows = 0
    fragmented_rows = 0

    for y in rows_y:
        row = mask[y]
        xs = np.where(row > 0)[0]

        if len(xs) < MIN_ROW_PIXELS:
            rejected_rows += 1
            continue

        fragment_count = count_row_fragments(row)

        if fragment_count > MAX_ROW_FRAGMENT_COUNT:
            fragmented_rows += 1
            rejected_rows += 1
            continue

        left_x = int(xs[0])
        right_x = int(xs[-1])
        width_px = right_x - left_x
        width_ratio = width_px / w

        if width_ratio < MIN_ROAD_WIDTH_RATIO or width_ratio > MAX_ROAD_WIDTH_RATIO:
            rejected_rows += 1
            continue

        center_x = int((left_x + right_x) / 2)

        left_points.append((left_x, y))
        right_points.append((right_x, y))
        center_points.append((center_x, y))
        row_widths.append(width_px)

    return {
        "left_points_raw": left_points,
        "right_points_raw": right_points,
        "center_points_raw": center_points,
        "row_widths": row_widths,
        "valid_rows": len(center_points),
        "rejected_rows": rejected_rows,
        "fragmented_rows": fragmented_rows,
    }


# ============================================================
# CURVE FITTING
# ============================================================

def fit_curve(points):
    if len(points) < MIN_POINTS_FOR_FIT:
        return []

    pts = np.array(points, dtype=np.float32)

    xs = pts[:, 0]
    ys = pts[:, 1]

    degree = min(POLY_DEGREE, len(points) - 1)

    try:
        coeffs = np.polyfit(ys, xs, degree)
    except np.linalg.LinAlgError:
        return []

    y_fit = np.linspace(min(ys), max(ys), len(points)).astype(np.float32)
    x_fit = np.polyval(coeffs, y_fit)

    return [(int(x), int(y)) for x, y in zip(x_fit, y_fit)]


def geometry_from_curves(left_curve, right_curve, center_curve):
    if not center_curve:
        return {
            "mean_left_x": None,
            "mean_right_x": None,
            "mean_center_x": None,
            "mean_width_px": None,
        }

    mean_center_x = float(np.mean([p[0] for p in center_curve]))

    if left_curve and right_curve:
        mean_left_x = float(np.mean([p[0] for p in left_curve]))
        mean_right_x = float(np.mean([p[0] for p in right_curve]))
        mean_width_px = mean_right_x - mean_left_x
    else:
        mean_left_x = None
        mean_right_x = None
        mean_width_px = None

    return {
        "mean_left_x": mean_left_x,
        "mean_right_x": mean_right_x,
        "mean_center_x": mean_center_x,
        "mean_width_px": mean_width_px,
    }


# ============================================================
# TEMPORAL SMOOTHING
# ============================================================

def smooth_value(current, previous):
    if current is None:
        return previous

    if previous is None:
        return current

    return TEMPORAL_ALPHA * previous + (1.0 - TEMPORAL_ALPHA) * current


def smooth_geometry(current_geom, previous_geom):
    if previous_geom is None:
        return current_geom

    return {
        "mean_left_x": smooth_value(current_geom.get("mean_left_x"), previous_geom.get("mean_left_x")),
        "mean_right_x": smooth_value(current_geom.get("mean_right_x"), previous_geom.get("mean_right_x")),
        "mean_center_x": smooth_value(current_geom.get("mean_center_x"), previous_geom.get("mean_center_x")),
        "mean_width_px": smooth_value(current_geom.get("mean_width_px"), previous_geom.get("mean_width_px")),
    }


# ============================================================
# CONFIDENCE + FLAGS
# ============================================================

def calculate_confidence(mask, boundary_data, geometry, previous_geometry, width, height,
                         sample_quality_value, fragmentation_score, bottom_metrics):
    roi_area = width * height * (ROI_BOTTOM_RATIO - ROI_TOP_RATIO)
    mask_area_ratio = float(np.count_nonzero(mask)) / max(roi_area, 1)

    valid_row_ratio = boundary_data["valid_rows"] / max(BOUNDARY_ROWS, 1)
    fragmented_row_ratio = boundary_data["fragmented_rows"] / max(BOUNDARY_ROWS, 1)

    mean_width = geometry.get("mean_width_px")
    mean_center = geometry.get("mean_center_x")

    if mean_width is None:
        width_ratio = 0.0
        width_score = 0.0
    else:
        width_ratio = mean_width / width
        if MIN_ROAD_WIDTH_RATIO <= width_ratio <= MAX_ROAD_WIDTH_RATIO:
            width_score = 1.0
        else:
            width_score = 0.0

    if previous_geometry is not None and mean_center is not None and previous_geometry.get("mean_center_x") is not None:
        center_jump = abs(mean_center - previous_geometry["mean_center_x"]) / width
        center_stability_score = 1.0 - np.clip(center_jump / MAX_CENTER_JUMP_RATIO, 0.0, 1.0)
    else:
        center_jump = 0.0
        center_stability_score = 1.0

    # Area score punishes tiny/wrong masks much harder now.
    if mask_area_ratio < 0.10:
        area_score = mask_area_ratio / 0.10
    elif mask_area_ratio > 0.78:
        area_score = max(0.0, 1.0 - (mask_area_ratio - 0.78) / 0.22)
    else:
        area_score = 1.0

    bottom_area = bottom_metrics["bottom_area_ratio"]
    bottom_width = bottom_metrics["bottom_mean_width_ratio"]
    bottom_max_width = bottom_metrics["bottom_max_width_ratio"]

    bottom_area_score = np.clip(bottom_area / MIN_BOTTOM_AREA_RATIO, 0.0, 1.0)
    bottom_width_score = np.clip(bottom_width / MIN_BOTTOM_WIDTH_RATIO, 0.0, 1.0)

    fragmentation_quality = 1.0 - np.clip(fragmentation_score, 0.0, 1.0)
    row_fragment_quality = 1.0 - np.clip(fragmented_row_ratio, 0.0, 1.0)

    # Confidence now heavily rewards bottom road coverage and mask area,
    # because a tiny clean wrong blob should NOT get high confidence.
    confidence = (
        0.20 * valid_row_ratio
        + 0.20 * area_score
        + 0.20 * bottom_width_score
        + 0.15 * bottom_area_score
        + 0.10 * width_score
        + 0.07 * center_stability_score
        + 0.04 * fragmentation_quality
        + 0.02 * row_fragment_quality
        + 0.02 * sample_quality_value
    )

    confidence = float(np.clip(confidence, 0.0, 1.0))

    flags = []

    if mask_area_ratio < 0.10:
        flags.append("LOW_MASK_AREA")

    if mask_area_ratio > 0.78:
        flags.append("HIGH_MASK_AREA")

    if bottom_width < MIN_BOTTOM_WIDTH_RATIO:
        flags.append("BOTTOM_ROAD_TOO_NARROW")

    if bottom_area < MIN_BOTTOM_AREA_RATIO:
        flags.append("BOTTOM_ROAD_MISSING")

    if boundary_data["valid_rows"] < max(6, int(BOUNDARY_ROWS * 0.35)):
        flags.append("BOUNDARY_MISSING")

    if mean_width is None:
        flags.append("WIDTH_MISSING")
    elif width_ratio < MIN_ROAD_WIDTH_RATIO or width_ratio > MAX_ROAD_WIDTH_RATIO:
        flags.append("WIDTH_IMPLAUSIBLE")

    if fragmentation_score > 0.45 or fragmented_row_ratio > 0.30:
        flags.append("FRAGMENTED_MASK")

    if center_stability_score < 0.45:
        flags.append("CENTER_JUMP")

    if sample_quality_value < 0.55:
        flags.append("BAD_SAMPLE_PATCH")

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        flags.append("LOW_CONFIDENCE")

    metrics = {
        "mask_area_ratio": float(mask_area_ratio),
        "valid_row_ratio": float(valid_row_ratio),
        "fragmented_row_ratio": float(fragmented_row_ratio),
        "width_ratio": float(width_ratio),
        "width_score": float(width_score),
        "area_score": float(area_score),
        "bottom_area_ratio": float(bottom_area),
        "bottom_mean_width_ratio": float(bottom_width),
        "bottom_max_width_ratio": float(bottom_max_width),
        "bottom_area_score": float(bottom_area_score),
        "bottom_width_score": float(bottom_width_score),
        "center_stability_score": float(center_stability_score),
        "center_jump_ratio": float(center_jump),
        "fragmentation_score": float(fragmentation_score),
        "sample_quality": float(sample_quality_value),
    }

    return confidence, flags, metrics


# ============================================================
# DRAWING
# ============================================================

def overlay_mask(frame, mask):
    overlay = frame.copy()

    green = np.zeros_like(frame)
    green[:, :] = (0, 255, 0)

    mask_bool = mask > 0
    blended = cv2.addWeighted(frame, 1.0 - MASK_ALPHA, green, MASK_ALPHA, 0)
    overlay[mask_bool] = blended[mask_bool]

    return overlay


def draw_polyline(frame, points, color, thickness=3):
    if len(points) < 2:
        return

    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], False, color, thickness)


def draw_points(frame, points, color, radius=2):
    for x, y in points:
        cv2.circle(frame, (int(x), int(y)), radius, color, -1)


def draw_debug(frame, mask, boundary_data, curves, geometry, confidence, flags, metrics, frame_idx, time_s):
    debug = overlay_mask(frame, mask)

    draw_polyline(debug, curves["left"], (255, 0, 0), 3)
    draw_polyline(debug, curves["right"], (0, 0, 255), 3)
    draw_polyline(debug, curves["center"], (0, 255, 255), 3)

    draw_points(debug, boundary_data["left_points_raw"], (255, 80, 80), 2)
    draw_points(debug, boundary_data["right_points_raw"], (80, 80, 255), 2)
    draw_points(debug, boundary_data["center_points_raw"], (0, 255, 255), 2)

    flag_text = ",".join(flags) if flags else "NONE"

    lines = [
        f"Frame: {frame_idx}",
        f"Time: {time_s:.2f}s",
        f"Confidence: {confidence:.2f}",
        f"Flags: {flag_text}",
        f"Center X: {geometry.get('mean_center_x')}",
        f"Road width px: {geometry.get('mean_width_px')}",
        f"Mask area: {metrics['mask_area_ratio']:.2f}",
        f"Bottom width: {metrics['bottom_mean_width_ratio']:.2f}",
        f"Bottom area: {metrics['bottom_area_ratio']:.2f}",
        f"Valid rows: {metrics['valid_row_ratio']:.2f}",
    ]

    y = 35

    for text in lines:
        cv2.putText(
            debug,
            text,
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )
        y += 26

    return debug


# ============================================================
# MAIN
# ============================================================

def main():
    session_path = get_session_path()
    meta = load_meta(session_path)
    video_path = get_video_path(session_path, meta)

    processed_dir = session_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    debug_video_path = processed_dir / f"{session_path.name}_environment_debug.mp4"
    data_csv_path = processed_dir / f"{session_path.name}_environment_data.csv"
    summary_path = processed_dir / f"{session_path.name}_environment_summary.json"

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = float(meta.get("fps", 30))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(debug_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    roi_mask = make_roi_mask(height, width)

    rows = []
    confidence_values = []
    flag_counts = {}

    previous_geometry_raw = None
    previous_geometry_smooth = None

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        time_s = frame_idx / fps

        likelihood, sample_q = adaptive_road_likelihood(frame, roi_mask)
        raw_mask, threshold = likelihood_to_mask(likelihood)
        road_mask, component_count, fragmentation_score = clean_mask(raw_mask, width, height)

        bottom_metrics = bottom_road_metrics(road_mask)

        boundary_data = extract_boundaries(road_mask)

        left_curve = fit_curve(boundary_data["left_points_raw"])
        right_curve = fit_curve(boundary_data["right_points_raw"])
        center_curve = fit_curve(boundary_data["center_points_raw"])

        raw_geometry = geometry_from_curves(left_curve, right_curve, center_curve)
        smooth_geom = smooth_geometry(raw_geometry, previous_geometry_smooth)

        confidence, flags, metrics = calculate_confidence(
            road_mask,
            boundary_data,
            smooth_geom,
            previous_geometry_raw,
            width,
            height,
            sample_q,
            fragmentation_score,
            bottom_metrics
        )

        previous_geometry_raw = raw_geometry
        previous_geometry_smooth = smooth_geom

        confidence_values.append(confidence)

        for flag in flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

        curves = {
            "left": left_curve,
            "right": right_curve,
            "center": center_curve,
        }

        debug_frame = draw_debug(
            frame,
            road_mask,
            boundary_data,
            curves,
            smooth_geom,
            confidence,
            flags,
            metrics,
            frame_idx,
            time_s
        )

        writer.write(debug_frame)

        rows.append({
            "frame_idx": frame_idx,
            "time_s": time_s,

            "left_x": smooth_geom.get("mean_left_x"),
            "right_x": smooth_geom.get("mean_right_x"),
            "center_x": smooth_geom.get("mean_center_x"),
            "road_width_px": smooth_geom.get("mean_width_px"),

            "confidence": confidence,
            "failure_flags": "|".join(flags),

            "mask_area_ratio": metrics["mask_area_ratio"],
            "valid_row_ratio": metrics["valid_row_ratio"],
            "fragmented_row_ratio": metrics["fragmented_row_ratio"],
            "width_ratio": metrics["width_ratio"],
            "bottom_area_ratio": metrics["bottom_area_ratio"],
            "bottom_mean_width_ratio": metrics["bottom_mean_width_ratio"],
            "bottom_max_width_ratio": metrics["bottom_max_width_ratio"],
            "bottom_area_score": metrics["bottom_area_score"],
            "bottom_width_score": metrics["bottom_width_score"],
            "center_stability_score": metrics["center_stability_score"],
            "center_jump_ratio": metrics["center_jump_ratio"],
            "fragmentation_score": metrics["fragmentation_score"],
            "sample_quality": metrics["sample_quality"],

            "component_count": component_count,
            "mask_threshold": threshold,
            "valid_rows": boundary_data["valid_rows"],
            "rejected_rows": boundary_data["rejected_rows"],
            "fragmented_rows": boundary_data["fragmented_rows"],
        })

        frame_idx += 1

    cap.release()
    writer.release()

    df = pd.DataFrame(rows)
    df.to_csv(data_csv_path, index=False)

    good_frames = int((df["confidence"] >= GOOD_CONFIDENCE_THRESHOLD).sum()) if len(df) else 0
    low_frames = int((df["confidence"] < LOW_CONFIDENCE_THRESHOLD).sum()) if len(df) else 0

    summary = {
        "session": session_path.name,
        "video_path": str(video_path),

        "outputs": {
            "debug_video": str(debug_video_path),
            "environment_data_csv": str(data_csv_path),
            "summary_json": str(summary_path),
        },

        "video": {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
        },

        "results": {
            "processed_frames": int(len(df)),
            "mean_confidence": float(df["confidence"].mean()) if len(df) else None,
            "min_confidence": float(df["confidence"].min()) if len(df) else None,
            "max_confidence": float(df["confidence"].max()) if len(df) else None,
            "good_frame_count": good_frames,
            "good_frame_ratio": float(good_frames / len(df)) if len(df) else None,
            "low_confidence_frame_count": low_frames,
            "low_confidence_frame_ratio": float(low_frames / len(df)) if len(df) else None,
            "flag_counts": flag_counts,
        },

        "status": "ok",
        "notes": (
            "Updated baseline: wider sample patch, less strict threshold, stronger confidence penalties, "
            "bottom-road coverage checks, and protection against clean-but-wrong tiny masks."
        )
    }

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Environment debug video: {debug_video_path}")
    print(f"Environment CSV: {data_csv_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
