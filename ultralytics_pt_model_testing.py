import cv2
import time
from ultralytics import YOLO

# ---------------- CONFIG ----------------
video_path = "brt_presentation.mp4"
model_path = "yolo_nano_v2_1_class_640_no_filter.pt"   # <-- .pt model
YOLO_CONFIDENCE = 0.2
OUTPUT_VIDEO = "ultralytics_output.mp4"
# ----------------------------------------


def main():
    print("YOLOv8 PyTorch Inference Benchmark")
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    print(f"Confidence threshold: {YOLO_CONFIDENCE}")

    # Load YOLOv8 .pt model
    print("Loading YOLOv8 model (.pt)...")
    model = YOLO(model_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Total frames: {total_frames}")

    # Output video writer
    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 25,
        (width, height)
    )

    if not writer.isOpened():
        print("Error: Could not open output video for writing.")
        return

    frame_count = 0
    inference_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Time YOLO inference
        start = time.perf_counter()
        results = model(frame, conf=YOLO_CONFIDENCE, verbose=False)
        elapsed = time.perf_counter() - start
        inference_times.append(elapsed)

        # Draw bounding boxes
        annotated = frame.copy()
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0]) if hasattr(box, "cls") else 0

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cls} {conf:.2f}"
                cv2.putText(annotated, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        writer.write(annotated)

        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()
    writer.release()

    if inference_times:
        avg_time = sum(inference_times) / len(inference_times)
        print("\n" + "="*50)
        print(f"Total frames processed: {frame_count}")
        print(f"Average inference time: {avg_time*1000:.2f} ms")
        print(f"FPS (theoretical): {1.0/avg_time:.2f}")
        print(f"Output saved as: {OUTPUT_VIDEO}")
        print("="*50)
    else:
        print("No frames processed!")


if __name__ == "__main__":
    main()
