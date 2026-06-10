import csv
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from detection_pipeline import (
        load_video,
        read_frame,
        get_video_properties,
        create_video_writer,
        print_progress
)


def load_road_segmentation_model():
        model_name = "nvidia/segformer-b0-finetuned-cityscapes-768-768"

        processor = AutoImageProcessor.from_pretrained(model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)

        model.eval()

        return processor, model


def run_road_segmentation(processor, model, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inputs = processor(
                images=rgb_frame,
                return_tensors="pt"
        )

        with torch.no_grad():
                outputs = model(**inputs)

        logits = outputs.logits

        upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=frame.shape[:2],
                mode="bilinear",
                align_corners=False
        )

        segmentation = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        return segmentation


def create_road_mask(segmentation, model):
        id_to_label = model.config.id2label

        road_ids = []

        for class_id, label_name in id_to_label.items():
                if label_name.lower() == "road":
                        road_ids.append(int(class_id))

        road_mask = np.isin(segmentation, road_ids).astype(np.uint8)

        return road_mask


def create_drivable_mask(segmentation, model):
        id_to_label = model.config.id2label

        drivable_labels = [
                "road"
        ]

        drivable_ids = []

        for class_id, label_name in id_to_label.items():
                if label_name.lower() in drivable_labels:
                        drivable_ids.append(int(class_id))

        drivable_mask = np.isin(segmentation, drivable_ids).astype(np.uint8)

        return drivable_mask


def create_debug_frame(frame, road_mask, drivable_mask):
        overlay = frame.copy()

        road_color = np.zeros_like(frame)
        road_color[:, :, 1] = road_mask * 255

        overlay = cv2.addWeighted(
                overlay,
                0.65,
                road_color,
                0.35,
                0
        )

        return overlay


def extract_road_stats(frame_idx, time_s, road_mask, drivable_mask, width, height):
        frame_area = width * height

        road_area = int(road_mask.sum())
        drivable_area = int(drivable_mask.sum())

        road_area_ratio = road_area / frame_area
        drivable_area_ratio = drivable_area / frame_area

        row = [
                frame_idx,
                time_s,
                road_area,
                road_area_ratio,
                drivable_area,
                drivable_area_ratio
        ]

        return row


def create_road_csv_writer(csv_path):
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)

        writer.writerow([
                "frame",
                "time_s",
                "road_area",
                "road_area_ratio",
                "drivable_area",
                "drivable_area_ratio"
        ])

        return csv_file, writer


def main():
        if len(sys.argv) < 2:
                print("Usage:")
                print("python3 src/road_segmentation_pipeline.py session_020")
                return

        session_name = sys.argv[1]

        session_dir = Path("data") / "sessions" / session_name
        raw_dir = session_dir / "raw"
        processed_dir = session_dir / "processed"

        processed_dir.mkdir(parents=True, exist_ok=True)

        input_video_path = raw_dir / f"{session_name}_video.mp4"
        output_video_path = processed_dir / f"{session_name}_semantic_seg.mp4"
        output_csv_path = processed_dir / f"{session_name}_semantic_seg.csv"

        processor, model = load_road_segmentation_model()

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

        csv_file, csv_writer = create_road_csv_writer(output_csv_path)

        frame_idx = 0

        while True:
                success, frame = read_frame(video)

                if not success:
                        break

                time_s = frame_idx / fps

                segmentation = run_road_segmentation(
                        processor,
                        model,
                        frame
                )

                road_mask = create_road_mask(
                        segmentation,
                        model
                )

                drivable_mask = create_drivable_mask(
                        segmentation,
                        model
                )

                row = extract_road_stats(
                        frame_idx,
                        time_s,
                        road_mask,
                        drivable_mask,
                        width,
                        height
                )

                csv_writer.writerow(row)

                debug_frame = create_debug_frame(
                        frame,
                        road_mask,
                        drivable_mask
                )

                writer.write(debug_frame)

                frame_idx += 1

                if frame_idx % 100 == 0:
                        print_progress(frame_idx, total_frames)

        print()

        video.release()
        writer.release()
        csv_file.close()

        print("Semantic segmentation video saved:")
        print(output_video_path)

        print("Semantic segmentation CSV saved:")
        print(output_csv_path)


if __name__ == "__main__":
        main()