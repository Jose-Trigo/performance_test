import cv2
import time
import numpy as np
from openvino.runtime import Core

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
video_path = "brt_presentation.mp4"
model_dir = "yolo_nano_v2_1_class_640_no_filter_int8_openvino_model"
YOLO_CONFIDENCE = 0.2
IOU_THRESHOLD = 0.45
OUTPUT_VIDEO = "output.mp4"

# ---------------------------------------------------------
# YOLO postprocessing
# ---------------------------------------------------------
def nms(boxes, scores, iou_threshold):
    idxs = scores.argsort()[::-1]
    keep = []

    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)

        if len(idxs) == 1:
            break

        xx1 = np.maximum(boxes[current, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[current, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[current, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[current, 3], boxes[idxs[1:], 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        area1 = (boxes[current, 2] - boxes[current, 0]) * (boxes[current, 3] - boxes[current, 1])
        area2 = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])

        iou = inter / (area1 + area2 - inter + 1e-6)
        idxs = idxs[1:][iou < iou_threshold]

    return keep

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    print("OpenVINO YOLO Inference Benchmark")
    print(f"Model directory: {model_dir}")
    print(f"Video: {video_path}")

    core = Core()
    model = core.read_model(f"{model_dir}/yolo_nano_v2_1_class_640_no_filter_int8.xml")
    compiled_model = core.compile_model(model, "CPU")

    input_layer = compiled_model.inputs[0]
    output_layer = compiled_model.outputs[0]

    _, _, input_h, input_w = input_layer.shape

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

        # ---------------- POSTPROCESS ----------------
        # Output shape: (1, 5, 8400) -> squeeze to (5, 8400)
        # Row 0: x, Row 1: y, Row 2: w, Row 3: h, Row 4: score
        preds = outputs.squeeze()
        
        x_vals = preds[0, :]
        y_vals = preds[1, :]
        w_vals = preds[2, :]
        h_vals = preds[3, :]
        scores = preds[4, :]

        # Filter by confidence
        mask = scores >= YOLO_CONFIDENCE
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]
        w_vals = w_vals[mask]
        h_vals = h_vals[mask]
        scores = scores[mask]

        if len(scores) > 0:
            # Scale coordinates from 640x640 to original frame size
            cx = x_vals * img_w / 640.0
            cy = y_vals * img_h / 640.0
            bw = w_vals * img_w / 640.0
            bh = h_vals * img_h / 640.0

            # Convert from center format to corner format (x1, y1, x2, y2)
            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = cx + bw / 2.0
            y2 = cy + bh / 2.0

            boxes = np.stack([x1, y1, x2, y2], axis=1)

            # NMS
            keep = nms(boxes, scores, IOU_THRESHOLD)
            boxes = boxes[keep]
            scores = scores[keep]
        else:
            boxes = np.empty((0, 4))
            scores = np.empty((0,))

        total_time = time.perf_counter() - start_total
        inference_times.append(total_time)

        # ---------------- DRAW + WRITE (NOT TIMED) ----------------
        annotated = frame.copy()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, f"{score:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        writer.write(annotated)

        if frame_count % 100 == 0:
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