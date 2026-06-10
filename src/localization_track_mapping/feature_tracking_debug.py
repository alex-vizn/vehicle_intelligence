import sys
from pathlib import Path

import cv2
import numpy as np


def load_video(video_path):
        video = cv2.VideoCapture(str(video_path))
        return video


def create_video_writer(output_path, fps, width, height):
        writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
        )
        return writer


def detect_features(gray_frame):
        features = cv2.goodFeaturesToTrack(
                gray_frame,
                maxCorners=300,
                qualityLevel=0.01,
                minDistance=10,
                blockSize=7
        )

        return features


def track_features(prev_gray, gray, prev_features):
        next_features, status, error = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                prev_features,
                None
        )

        return next_features, status, error


def estimate_motion(prev_features, next_features, status):
        good_prev = prev_features[status.flatten() == 1]
        good_next = next_features[status.flatten() == 1]

        if len(good_prev) < 20:
                return 0.0, 0.0

        movement = good_next - good_prev
        average_movement = movement.mean(axis=0).ravel()

        dx = -float(average_movement[0])
        dy = -float(average_movement[1])

        return dx, dy


def draw_tracks(frame, prev_features, next_features, status):
        output = frame.copy()

        for i in range(len(status)):
                if status[i] == 1:
                        x1, y1 = prev_features[i].ravel()
                        x2, y2 = next_features[i].ravel()

                        cv2.circle(
                                output,
                                (int(x2), int(y2)),
                                3,
                                (0, 255, 0),
                                -1
                        )

                        cv2.line(
                                output,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (255, 0, 0),
                                1
                        )

        return output


def draw_trajectory_canvas(trajectory, width=500, height=500):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        if len(trajectory) < 2:
                return canvas

        points = np.array(trajectory)

        min_x = points[:, 0].min()
        max_x = points[:, 0].max()
        min_y = points[:, 1].min()
        max_y = points[:, 1].max()

        scale_x = width / max(max_x - min_x, 1)
        scale_y = height / max(max_y - min_y, 1)
        scale = min(scale_x, scale_y) * 0.8

        center_x = width // 2
        center_y = height // 2

        mean_x = points[:, 0].mean()
        mean_y = points[:, 1].mean()

        draw_points = []

        for x, y in trajectory:
                px = int(center_x + (x - mean_x) * scale)
                py = int(center_y + (y - mean_y) * scale)

                draw_points.append((px, py))

        for i in range(1, len(draw_points)):
                cv2.line(
                        canvas,
                        draw_points[i - 1],
                        draw_points[i],
                        (0, 255, 0),
                        2
                )

        cv2.circle(
                canvas,
                draw_points[-1],
                5,
                (0, 0, 255),
                -1
        )

        cv2.putText(
                canvas,
                "Estimated trajectory",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
        )

        return canvas


def combine_video_and_trajectory(video_frame, trajectory_canvas):
        video_height = video_frame.shape[0]

        trajectory_canvas = cv2.resize(
                trajectory_canvas,
                (trajectory_canvas.shape[1], video_height)
        )

        combined = np.hstack([
                video_frame,
                trajectory_canvas
        ])

        return combined


def main():
        if len(sys.argv) < 2:
                print("Usage:")
                print("PYTHONPATH=src python3 src/localization_track_mapping/feature_tracking_debug.py session_020")
                return

        session_name = sys.argv[1]

        session_path = Path("data") / "sessions" / session_name
        video_path = session_path / "raw" / f"{session_name}_video.mp4"

        processed_dir = session_path / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        output_video_path = processed_dir / f"{session_name}_feature_tracking_debug.mp4"

        video = load_video(video_path)

        if not video.isOpened():
                print("Could not open video:")
                print(video_path)
                return

        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_width = width + 500

        writer = create_video_writer(
                output_video_path,
                fps,
                output_width,
                height
        )

        success, prev_frame = video.read()

        if not success:
                print("Could not read first frame")
                return

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_features = detect_features(prev_gray)

        x = 0.0
        y = 0.0
        trajectory = [(x, y)]

        frame_idx = 1

        while True:
                success, frame = video.read()

                if not success:
                        break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_features is None or len(prev_features) < 20:
                        prev_features = detect_features(prev_gray)

                next_features, status, error = track_features(
                        prev_gray,
                        gray,
                        prev_features
                )

                dx, dy = estimate_motion(
                        prev_features,
                        next_features,
                        status
                )

                x += dx
                y += dy

                trajectory.append((x, y))

                debug_frame = draw_tracks(
                        frame,
                        prev_features,
                        next_features,
                        status
                )

                trajectory_canvas = draw_trajectory_canvas(trajectory)

                combined_frame = combine_video_and_trajectory(
                        debug_frame,
                        trajectory_canvas
                )

                writer.write(combined_frame)

                good_count = int(status.sum())

                if frame_idx % 100 == 0:
                        print(f"\rFrame {frame_idx} | tracked features: {good_count}", end="", flush=True)

                if good_count < 80:
                        prev_features = detect_features(gray)
                else:
                        prev_features = next_features[status == 1].reshape(-1, 1, 2)

                prev_gray = gray
                frame_idx += 1

        print()

        video.release()
        writer.release()

        print("Feature tracking debug saved:")
        print(output_video_path)


if __name__ == "__main__":
        main()

