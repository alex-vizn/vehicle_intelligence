import csv
import json
import sys
from pathlib import Path
from ultralytics import YOLO

from detection_pipeline import (
        load_video,
        read_frame,
        get_video_properties,
        create_video_writer,
        print_progress
)


def load_tracker():
        tracker = YOLO("yolo11n.pt")
        return tracker


def run_tracking(tracker, frame):
        results = tracker.track(
                frame,
                persist=True,
                verbose=False
        )
        return results


def extract_tracks(results, frame_idx, time_s, min_confidence=0.50):
        rows = []

        boxes = results[0].boxes

        if boxes.id is None:
                return rows

        for box in boxes:
                track_id = int(box.id[0])
                class_id = int(box.cls[0])
                class_name = results[0].names[class_id]
                confidence = float(box.conf[0])

                if confidence < min_confidence:
                        continue

                x1, y1, x2, y2 = box.xyxy[0]

                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                rows.append([
                        frame_idx,
                        time_s,
                        track_id,
                        class_name,
                        confidence,
                        x1,
                        y1,
                        x2,
                        y2,
                        center_x,
                        center_y
                ])

        return rows


def create_track_csv_writer(csv_path):
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)

        writer.writerow([
                "frame",
                "time_s",
                "track_id",
                "class",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "center_x",
                "center_y"
        ])

        return csv_file, writer


def update_track_stats(track_stats, rows):
        for row in rows:
                track_id = row[2]
                class_name = row[3]

                if track_id not in track_stats:
                        track_stats[track_id] = {
                                "class": class_name,
                                "frames_seen": 0
                        }

                track_stats[track_id]["frames_seen"] += 1


def save_tracking_summary(summary_path, track_stats):
        total_tracks = len(track_stats)

        if total_tracks > 0:
                track_lengths = [
                        data["frames_seen"]
                        for data in track_stats.values()
                ]

                longest_track = max(track_lengths)
                average_track_length = sum(track_lengths) / total_tracks
        else:
                longest_track = 0
                average_track_length = 0

        summary = {
                "total_tracks": total_tracks,
                "longest_track_frames": longest_track,
                "average_track_length_frames": average_track_length
        }

        with open(summary_path, "w") as file:
                json.dump(summary, file, indent=4)


def main():
        if len(sys.argv) < 2:
                print("Usage:")
                print("python3 src/tracking_pipeline.py session_020")
                return

        session_name = sys.argv[1]

        session_dir = Path("data") / "sessions" / session_name
        raw_dir = session_dir / "raw"
        processed_dir = session_dir / "processed"

        processed_dir.mkdir(parents=True, exist_ok=True)

        input_video_path = raw_dir / f"{session_name}_video.mp4"
        output_video_path = processed_dir / f"{session_name}_tracking_debug.mp4"
        output_csv_path = processed_dir / f"{session_name}_tracks.csv"
        output_summary_path = processed_dir / f"{session_name}_tracking_summary.json"

        tracker = load_tracker()

        video = load_video(input_video_path)

        if not video.isOpened():
                print("Could not open video:")
                print(input_video_path)
                return

        fps, width, height, total_frames = get_video_properties(video)

        writer = create_video_writer(
                output_video_path,
                fps,
                width,
                height
        )

        csv_file, csv_writer = create_track_csv_writer(output_csv_path)

        track_stats = {}

        frame_idx = 0

        while True:
                success, frame = read_frame(video)

                if not success:
                        break

                time_s = frame_idx / fps

                results = run_tracking(tracker, frame)

                rows = extract_tracks(
                        results,
                        frame_idx,
                        time_s
                )

                for row in rows:
                        csv_writer.writerow(row)

                update_track_stats(track_stats, rows)

                annotated_frame = results[0].plot()
                writer.write(annotated_frame)

                frame_idx += 1

                if frame_idx % 100 == 0:
                        print_progress(frame_idx, total_frames)

        print()

        video.release()
        writer.release()
        csv_file.close()

        save_tracking_summary(
                output_summary_path,
                track_stats
        )

        print("Tracking video saved:")
        print(output_video_path)

        print("Tracking CSV saved:")
        print(output_csv_path)

        print("Tracking summary saved:")
        print(output_summary_path)


if __name__ == "__main__":
        main()