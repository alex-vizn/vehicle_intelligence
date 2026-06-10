import csv
import sys
from pathlib import Path

from detection_pipeline import (
        load_video,
        read_frame,
        get_video_properties,
        print_progress
)



def load_csv_by_frame(csv_path):
        rows_by_frame = {}

        if not csv_path.exists():
                return rows_by_frame

        with open(csv_path, "r") as file:
                reader = csv.DictReader(file)

                for row in reader:
                        frame = int(float(row["frame"]))

                        if frame not in rows_by_frame:
                                rows_by_frame[frame] = []

                        rows_by_frame[frame].append(row)

        return rows_by_frame


# New function inserted here
def load_motion_by_frame(csv_path, fps):
        rows_by_frame = {}

        if not csv_path.exists():
                return rows_by_frame

        with open(csv_path, "r") as file:
                reader = csv.DictReader(file)

                for row in reader:
                        start_s = float(row["start_s"])
                        end_s = float(row["end_s"])

                        start_frame = int(start_s * fps)
                        end_frame = int(end_s * fps)

                        for frame in range(start_frame, end_frame + 1):
                                if frame not in rows_by_frame:
                                        rows_by_frame[frame] = []

                                rows_by_frame[frame].append(row)

        return rows_by_frame


def get_motion_label(motion_rows):
        if len(motion_rows) == 0:
                return "unknown"

        row = motion_rows[0]

        possible_columns = [
                "motion_label",
                "motion_state",
                "label",
                "state",
                "event"
        ]

        for column in possible_columns:
                if column in row:
                        return row[column]

        return "unknown"


def get_main_track(track_rows):
        if len(track_rows) == 0:
                return {
                        "track_id": "",
                        "class": "",
                        "confidence": "",
                        "center_x": "",
                        "center_y": ""
                }

        best_track = max(
                track_rows,
                key=lambda row: float(row["confidence"])
        )

        return {
                "track_id": best_track["track_id"],
                "class": best_track["class"],
                "confidence": best_track["confidence"],
                "center_x": best_track["center_x"],
                "center_y": best_track["center_y"]
        }


def count_class(rows, class_name):
        count = 0

        for row in rows:
                if row["class"] == class_name:
                        count += 1

        return count


def create_scene_row(frame_idx, time_s, detection_rows, track_rows, road_rows, motion_rows):
        if len(road_rows) > 0:
                road_area_ratio = float(road_rows[0]["road_area_ratio"])
                drivable_area_ratio = float(road_rows[0]["drivable_area_ratio"])
        else:
                road_area_ratio = 0.0
                drivable_area_ratio = 0.0

        road_visible = road_area_ratio > 0.10

        num_detections = len(detection_rows)
        num_tracks = len(track_rows)

        num_cars = count_class(detection_rows, "car")
        num_people = count_class(detection_rows, "person")

        main_track = get_main_track(track_rows)

        motion_label = get_motion_label(motion_rows)

        if not road_visible and num_tracks == 0:
                scene_quality = "low_information"
        elif not road_visible:
                scene_quality = "weak_road"
        elif num_tracks == 0:
                scene_quality = "no_tracks"
        else:
                scene_quality = "good"

        row = [
                frame_idx,
                time_s,
                road_area_ratio,
                drivable_area_ratio,
                road_visible,
                num_detections,
                num_tracks,
                num_cars,
                num_people,
                main_track["track_id"],
                main_track["class"],
                main_track["confidence"],
                main_track["center_x"],
                main_track["center_y"],
                motion_label,
                scene_quality
        ]

        return row


def create_scene_csv_writer(csv_path):
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)

        writer.writerow([
                "frame",
                "time_s",
                "road_area_ratio",
                "drivable_area_ratio",
                "road_visible",
                "num_detections",
                "num_tracks",
                "num_cars",
                "num_people",
                "main_track_id",
                "main_track_class",
                "main_track_confidence",
                "main_track_center_x",
                "main_track_center_y",
                "motion_label",
                "scene_quality"
        ])

        return csv_file, writer


def main():
        if len(sys.argv) < 2:
                print("Usage:")
                print("python3 src/scene_representation.py session_020")
                return

        session_name = sys.argv[1]

        session_dir = Path("data") / "sessions" / session_name
        raw_dir = session_dir / "raw"
        processed_dir = session_dir / "processed"

        input_video_path = raw_dir / f"{session_name}_video.mp4"

        detections_csv_path = processed_dir / f"{session_name}_detections.csv"
        tracks_csv_path = processed_dir / f"{session_name}_tracks.csv"
        road_csv_path = processed_dir / f"{session_name}_semantic_seg.csv"
        motion_csv_path = processed_dir / f"{session_name}_motion_timeline.csv"

        output_csv_path = processed_dir / f"{session_name}_scene_state.csv"

        detections_by_frame = load_csv_by_frame(detections_csv_path)
        tracks_by_frame = load_csv_by_frame(tracks_csv_path)
        road_by_frame = load_csv_by_frame(road_csv_path)

        video = load_video(input_video_path)

        if not video.isOpened():
                print("Could not open video:")
                print(input_video_path)
                return

        fps, width, height, total_frames = get_video_properties(video)

        motion_by_frame = load_motion_by_frame(motion_csv_path, fps)

        csv_file, csv_writer = create_scene_csv_writer(output_csv_path)

        frame_idx = 0

        while True:
                success, frame = read_frame(video)

                if not success:
                        break

                time_s = frame_idx / fps

                detection_rows = detections_by_frame.get(frame_idx, [])
                track_rows = tracks_by_frame.get(frame_idx, [])
                road_rows = road_by_frame.get(frame_idx, [])
                motion_rows = motion_by_frame.get(frame_idx, [])

                scene_row = create_scene_row(
                        frame_idx,
                        time_s,
                        detection_rows,
                        track_rows,
                        road_rows,
                        motion_rows
                )

                csv_writer.writerow(scene_row)

                frame_idx += 1

                if frame_idx % 100 == 0:
                        print_progress(frame_idx, total_frames)

        print()

        video.release()
        csv_file.close()

        print("Scene state CSV saved:")
        print(output_csv_path)


if __name__ == "__main__":
        main()

