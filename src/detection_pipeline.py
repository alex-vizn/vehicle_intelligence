import cv2
import csv
import sys
from pathlib import Path
from ultralytics import YOLO


def load_detector():
        model = YOLO("yolo11n.pt")
        return model


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


def get_video_properties(video):
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        return fps, width, height, total_frames


def read_frame(video):
        success, frame = video.read()
        return success, frame


def print_progress(frame_idx, total_frames):
        print(
                f"\rProcessed frame {frame_idx}/{total_frames}",
                end="",
                flush=True
        )


def run_detection(model, frame):
        results = model(frame, verbose=False)
        return results


def extract_detections(results, frame_idx, time_s):
        rows = []

        boxes = results[0].boxes

        for box in boxes:
                class_id = int(box.cls[0])
                class_name = results[0].names[class_id]
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0]

                rows.append([
                        frame_idx,
                        time_s,
                        class_name,
                        confidence,
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2)
                ])

        return rows


def create_csv_writer(csv_path):
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)

        writer.writerow([
                "frame",
                "time_s",
                "class",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2"
        ])

        return csv_file, writer


def print_summary(csv_path):
        print()
        print("Detection CSV:")
        print(csv_path)


def main():
        if len(sys.argv) < 2:
                print("Usage:")
                print("python3 src/detection_pipeline.py session_020")
                return

        session_name = sys.argv[1]

        session_dir = Path("data") / "sessions" / session_name
        raw_dir = session_dir / "raw"
        processed_dir = session_dir / "processed"

        processed_dir.mkdir(parents=True, exist_ok=True)

        input_video_path = raw_dir / f"{session_name}_video.mp4"
        output_video_path = processed_dir / f"{session_name}_yolo_debug.mp4"
        output_csv_path = processed_dir / f"{session_name}_detections.csv"

        model = load_detector()

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

        csv_file, csv_writer = create_csv_writer(output_csv_path)

        frame_idx = 0

        while True:
                success, frame = read_frame(video)

                if not success:
                        break

                time_s = frame_idx / fps

                results = run_detection(model, frame)

                rows = extract_detections(
                        results,
                        frame_idx,
                        time_s
                )

                for row in rows:
                        csv_writer.writerow(row)

                annotated_frame = results[0].plot()
                writer.write(annotated_frame)

                frame_idx += 1

                if frame_idx % 100 == 0:
                        print_progress(frame_idx, total_frames)

        print()

        video.release()
        writer.release()
        csv_file.close()

        print("Detection video saved:")
        print(output_video_path)

        print("Detection CSV saved:")
        print(output_csv_path)

        print_summary(output_csv_path)


if __name__ == "__main__":
        main()