import cv2
import time
from ultralytics import YOLO

# Configuration
video_path = "brt_presentation.mp4"
#model_path = "models/yolo_nano_v2_1_class_640_no_filter_int8.tflite"
model_path = "models/yolo_nano_v2_1_class_640_no_filter_int8_openvino_model"
YOLO_CONFIDENCE = 0.2

def main():
    print("YOLO Inference Benchmark")
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    print(f"Confidence threshold: {YOLO_CONFIDENCE}")

    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(model_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    frame_count = 0
    inference_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Time YOLO inference
        start = time.perf_counter()
        _ = model(frame, verbose=False, conf=YOLO_CONFIDENCE,device="intel:gpu")
        elapsed = time.perf_counter() - start
        inference_times.append(elapsed)

        # Print progress every 100 frames
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()

    if inference_times:
        avg_time = sum(inference_times) / len(inference_times)
        print("\n" + "="*50)
        print(f"Total frames processed: {frame_count}")
        print(f"Average YOLO inference time: {avg_time*1000:.2f} ms")
        print(f"Theoretical FPS: {1.0/avg_time:.2f}")
        print("="*50)
    else:
        print("No frames processed!")

if __name__ == "__main__":
    main()