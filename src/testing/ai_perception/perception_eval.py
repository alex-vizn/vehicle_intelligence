import csv
import sys
from pathlib import Path

import cv2

from detection_pipeline import (
        load_video,
        read_frame,
        get_video_properties,
        print_progress
)


def load_scene_state(csv_path):
        rows_by_frame = {}

        with open(csv_path, "r") as file:
                reader = csv.DictReader(file)

                for row in reader:
                        frame = int(float(row["frame"]))
                        rows_by_frame[frame] = row

        return rows_by_frame


def create_failure_report_writer(csv_path):
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)

        writer.writerow([
                "frame",
                "time_s",
                "failure_type",
                "reason",
                "image_path"
        ])

        return csv_file, writer


def should_save_failure(failure_type, frame_idx, saved_counts, last_saved_frame, max_per_type, min_frame_gap):
        if saved_counts.get(failure_type, 0) >= max_per_type:
                return False

        if failure_type in last_saved_frame:
                if frame_idx - last_saved_frame[failure_type] < min_frame_gap:
                        return False

        return True


def find_failures(scene_row):
        failures = []

        road_area_ratio = float(scene_row["road_area_ratio"])
        drivable_area_ratio = float(scene_row["drivable_area_ratio"])
        road_visible = scene_row["road_visible"] == "True"
        num_detections = int(scene_row["num_detections"])
        num_tracks = int(scene_row["num_tracks"])
        scene_quality = scene_row["scene_quality"]

        main_track_confidence = scene_row["main_track_confidence"]

        if not road_visible:
                failures.append([
                        "no_road_visible",
                        "road_visible is False"
                ])

        if road_area_ratio < 0.10:
                failures.append([
                        "low_road_coverage",
                        "road_area_ratio below 0.10"
                ])

        if road_area_ratio > 0.80:
                failures.append([
                        "high_road_coverage",
                        "road_area_ratio above 0.80"
                ])

        if drivable_area_ratio < 0.10:
                failures.append([
                        "low_drivable_area",
                        "drivable_area_ratio below 0.10"
                ])

        if num_tracks == 0:
                failures.append([
                        "no_tracks",
                        "no tracked objects in frame"
                ])

        if num_detections == 0:
                failures.append([
                        "no_detections",
                        "no detected objects in frame"
                ])

        if num_detections > 15:
                failures.append([
                        "too_many_detections",
                        "more than 15 detections"
                ])

        if main_track_confidence != "":
                if float(main_track_confidence) < 0.55:
                        failures.append([
                                "low_main_track_confidence",
                                "main track confidence below 0.55"
                        ])

        if scene_quality != "good":
                failures.append([
                        "bad_scene_quality",
                        f"scene_quality is {scene_quality}"
                ])

        return failures


def save_failure_frame(frame, output_dir, failure_type, frame_idx):
        failure_dir = output_dir / failure_type
        failure_dir.mkdir(parents=True, exist_ok=True)

        image_path = failure_dir / f"frame_{frame_idx}.jpg"

        cv2.imwrite(str(image_path), frame)

        return image_path


def main():
        if len(sys.argv) < 2:
                print("Usage:")
                print("python3 src/perception_eval.py session_020")
                return

        session_name = sys.argv[1]

        session_dir = Path("data") / "sessions" / session_name
        raw_dir = session_dir / "raw"
        processed_dir = session_dir / "processed"

        input_video_path = raw_dir / f"{session_name}_video.mp4"
        scene_csv_path = processed_dir / f"{session_name}_scene_state.csv"

        review_dir = processed_dir / "review_frames"
        failure_report_path = processed_dir / "failure_report.csv"

        review_dir.mkdir(parents=True, exist_ok=True)

        scene_rows = load_scene_state(scene_csv_path)

        video = load_video(input_video_path)

        if not video.isOpened():
                print("Could not open video:")
                print(input_video_path)
                return

        fps, width, height, total_frames = get_video_properties(video)

        report_file, report_writer = create_failure_report_writer(failure_report_path)

        saved_counts = {}
        last_saved_frame = {}

        max_per_type = 50
        min_seconds_gap = 2
        min_frame_gap = int(fps * min_seconds_gap)

        frame_idx = 0

        while True:
                success, frame = read_frame(video)

                if not success:
                        break

                if frame_idx in scene_rows:
                        scene_row = scene_rows[frame_idx]

                        failures = find_failures(scene_row)

                        for failure in failures:
                                failure_type = failure[0]
                                reason = failure[1]

                                should_save = should_save_failure(
                                        failure_type,
                                        frame_idx,
                                        saved_counts,
                                        last_saved_frame,
                                        max_per_type,
                                        min_frame_gap
                                )

                                if should_save:
                                        image_path = save_failure_frame(
                                                frame,
                                                review_dir,
                                                failure_type,
                                                frame_idx
                                        )

                                        report_writer.writerow([
                                                frame_idx,
                                                scene_row["time_s"],
                                                failure_type,
                                                reason,
                                                image_path
                                        ])

                                        saved_counts[failure_type] = saved_counts.get(failure_type, 0) + 1
                                        last_saved_frame[failure_type] = frame_idx

                frame_idx += 1

                if frame_idx % 100 == 0:
                        print_progress(frame_idx, total_frames)

        print()

        video.release()
        report_file.close()

        print("Review frames saved:")
        print(review_dir)

        print("Failure report saved:")
        print(failure_report_path)


if __name__ == "__main__":
        main()