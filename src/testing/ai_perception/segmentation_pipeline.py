import csv
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


def load_segmentation_model():
        model = YOLO("yolo11n-seg.pt")
        return model


def run_segmentation(model, frame):
        results = model(frame, verbose=False)
        return results


def extract_segmentations(results, frame_idx, time_s, frame_width, frame_height, min_confidence=0.50):
        rows = []

        boxes = results[0].boxes
        masks = results[0].masks

        if masks is None:
                return rows

        for index, box in enumerate(boxes):
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

                mask = masks.data[index]

                mask_area = float(mask.sum())
                frame_area = frame_width * frame_height
                mask_area_ratio = mask_area / frame_area

                rows.append([
                        frame_idx,
                        time_s,
                        class_name,
                        confidence,
                        x1,
                        y1,
                        x2,
                        y2,
                        mask_area,
                        mask_area_ratio
                ])

        return rows


def create_segmentation_csv_writer(csv_path):
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
                "y2",
                "mask_area",
                "mask_area_ratio"
        ])

        return csv_file, writer


def main():
        if len(sys.argv) < 2:
                print("Usage:")
                print("python3 src/segmentation_pipeline.py session_020")
                return

        session_name = sys.argv[1]

        session_dir = Path("data") / "sessions" / session_name
        raw_dir = session_dir / "raw"
        processed_dir = session_dir / "processed"

        processed_dir.mkdir(parents=True, exist_ok=True)

        input_video_path = raw_dir / f"{session_name}_video.mp4"
        output_video_path = processed_dir / f"{session_name}_segmentation_debug.mp4"
        output_csv_path = processed_dir / f"{session_name}_segmentations.csv"

        model = load_segmentation_model()

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

        csv_file, csv_writer = create_segmentation_csv_writer(output_csv_path)

        frame_idx = 0

        while True:
                success, frame = read_frame(video)

                if not success:
                        break

                time_s = frame_idx / fps

                results = run_segmentation(model, frame)

                rows = extract_segmentations(
                        results,
                        frame_idx,
                        time_s,
                        width,
                        height
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

        print("Segmentation video saved:")
        print(output_video_path)

        print("Segmentation CSV saved:")
        print(output_csv_path)


if __name__ == "__main__":
        main()