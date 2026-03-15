import os
import time
import cv2
import numpy as np

#import tensorflow as tf                             # <-- WINDOWS

from tflite_runtime.interpreter import Interpreter   # <-- PI

# ---------------------------------------------------------
# Paths and config
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "yolo_nano_v2_1_class_640_no_filter_int8.tflite")

VIDEO_PATH = os.path.join(BASE_DIR, "brt_presentation.mp4")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "output_litert_int8.mp4")

YOLO_CONFIDENCE = 0.2
IOU_THRESHOLD = 0.45

INPUT_H = 640
INPUT_W = 640
OUTPUT_ELEMENTS = 1 * 5 * 8400  # (1, 5, 8400)

# ---------------------------------------------------------
# NMS
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
    print("TFLite Runtime YOLO Inference Benchmark")
    print(f"Model: {MODEL_PATH}")
    print(f"Video: {VIDEO_PATH}")

    # ---- NEW: TFLite Runtime model loading ----
    interpreter = Interpreter(model_path=MODEL_PATH, num_threads=4)   # PI
    #interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)         # WINDOWS
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    # --------------------------------------------

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Total frames: {total_frames}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")

    #writer = cv2.VideoWriter(
    #    OUTPUT_VIDEO,
    #    cv2.VideoWriter_fourcc(*"mp4v"),
    #    fps if fps > 0 else 25,
    #    (width, height)
    #)

    inference_times = []
    frame_count = 0
    debug_printed = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        img_h, img_w = frame.shape[:2]

        start_total = time.perf_counter()

        # PREPROCESS
        resized = cv2.resize(frame, (INPUT_W, INPUT_H))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = rgb.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # ---- NEW: TFLite inference ----
        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_index).reshape(5, 8400)
        # --------------------------------

        if not debug_printed:
            print("\n=== DEBUG: RAW MODEL OUTPUT ===")
            print("preds shape:", preds.shape)
            print("scores min/max:", preds[4].min(), preds[4].max())
            print("scores > 0.2:", np.sum(preds[4] > 0.2))
            print("================================\n")
            debug_printed = True

        x_vals = preds[0, :]
        y_vals = preds[1, :]
        w_vals = preds[2, :]
        h_vals = preds[3, :]
        scores = preds[4, :]

        mask = scores >= YOLO_CONFIDENCE
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]
        w_vals = w_vals[mask]
        h_vals = h_vals[mask]
        scores = scores[mask]

        if len(scores) > 0:
            cx = x_vals * img_w
            cy = y_vals * img_h
            bw = w_vals * img_w
            bh = h_vals * img_h

            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            boxes = np.stack([x1, y1, x2, y2], axis=1)

            keep = nms(boxes, scores, IOU_THRESHOLD)
            boxes = boxes[keep]
            scores = scores[keep]

        total_time = time.perf_counter() - start_total
        inference_times.append(total_time)

        annotated = frame.copy()
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, f"{score:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        #writer.write(annotated)

        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()
    #writer.release()

    if inference_times:
        avg = sum(inference_times) / len(inference_times)
        print("\n" + "="*50)
        print(f"Frames processed: {frame_count}")
        print(f"Average total time: {avg*1000:.2f} ms")
        print(f"FPS (end-to-end): {1.0/avg:.2f}")
        print("="*50)

if __name__ == "__main__":
    main()
