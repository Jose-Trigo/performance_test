import cv2
import time
import numpy as np
from openvino.runtime import Core

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
video_path = "brt_presentation.mp4"
model_dir = "models/yolo_26_nano_1_class_v1_no_filter_openvino_model"
YOLO_CONFIDENCE = 0.2
OUTPUT_VIDEO = "output_yolo26.mp4"

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    print("OpenVINO YOLO26 Inference (Decoded Output + Correct Scaling)")
    print(f"Model directory: {model_dir}")
    print(f"Video: {video_path}")

    core = Core()
    model = core.read_model(f"{model_dir}/yolo_26_nano_1_class_v1_no_filter.xml")
    compiled_model = core.compile_model(model, "CPU")

    input_layer = compiled_model.inputs[0]
    output_layer = compiled_model.outputs[0]

    _, _, input_h, input_w = input_layer.shape  # typically 640x640

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Total frames: {total_frames}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 25,
        (width, height)
    )

    if not writer.isOpened():
        print("Error: Could not open output.mp4 for writing.")
        return

    inference_times = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        img_h, img_w = frame.shape[:2]

        # ---------------- TOTAL TIMING ----------------
        start_total = time.perf_counter()

        # PREPROCESS
        resized = cv2.resize(frame, (input_w, input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = rgb.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None, ...]

        # INFERENCE
        outputs = compiled_model.infer_new_request({input_layer: img})[output_layer]

        # ---------------- PARSE DECODED YOLO26 OUTPUT ----------------
        # Shape: (1, 300, 6)
        preds = outputs[0]

        # Columns: x1, y1, x2, y2, score, class_id
        boxes = preds[:, :4]
        scores = preds[:, 4]
        classes = preds[:, 5].astype(int)

        # Filter by confidence
        mask = scores >= YOLO_CONFIDENCE
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]

        # ---------------- SCALE BOXES BACK TO ORIGINAL IMAGE ----------------
        scale_x = img_w / input_w   # original_width / 640
        scale_y = img_h / input_h   # original_height / 640

        annotated = frame.copy()

        for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
            # Scale from 640x640 → original frame size
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls} {score:.2f}"
            cv2.putText(
                annotated, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        writer.write(annotated)

        total_time = time.perf_counter() - start_total
        inference_times.append(total_time)

        if frame_count % 50 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()
    writer.release()

    if inference_times:
        avg = sum(inference_times) / len(inference_times)
        print("\n" + "="*50)
        print(f"Frames processed: {frame_count}")
        print(f"Average total time: {avg*1000:.2f} ms")
        print(f"FPS (end-to-end): {1.0/avg:.2f}")
        print(f"Annotated video saved as: {OUTPUT_VIDEO}")
        print("="*50)

if __name__ == "__main__":
    main()
