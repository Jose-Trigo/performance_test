import cv2
import time
import numpy as np
from openvino.runtime import Core, properties

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
video_path = "brt_presentation.mp4"
model_dir = "yolo_nano_v2_1_class_640_no_filter_openvino_model"
YOLO_CONFIDENCE = 0.2
IOU_THRESHOLD = 0.45

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

def main():
    print("OpenVINO YOLO Pure Inference Benchmark")
    core = Core()
    model = core.read_model(f"{model_dir}/yolo_nano_v2_1_class_640_no_filter.xml")
    # Try to use all logical cores (may be ignored on Windows)
    compiled_model = core.compile_model(
        model,
        "CPU",
        {
            properties.inference_num_threads(): 8,
            properties.hint.enable_hyper_threading(): True,
            properties.hint.scheduling_core_type(): properties.hint.SchedulingCoreType.ANY_CORE
        }
    )
    input_layer = compiled_model.inputs[0]
    output_layer = compiled_model.outputs[0]
    _, _, input_h, input_w = input_layer.shape

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    inference_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        img_h, img_w = frame.shape[:2]

        # ===== STEP 1: Preprocess =====
        preprocess_start = time.perf_counter()
        resized = cv2.resize(frame, (input_w, input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = rgb.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None, ...]
        preprocess_time = time.perf_counter() - preprocess_start

        # ===== STEP 2: Inference =====
        inference_start = time.perf_counter()
        outputs = compiled_model.infer_new_request({input_layer: img})[output_layer]
        inference_time = time.perf_counter() - inference_start

        # ===== STEP 3: Postprocess =====
        postprocess_start = time.perf_counter()
        preds = outputs.squeeze()
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
            cx = x_vals * img_w / 640.0
            cy = y_vals * img_h / 640.0
            bw = w_vals * img_w / 640.0
            bh = h_vals * img_h / 640.0
            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = cx + bw / 2.0
            y2 = cy + bh / 2.0
            boxes = np.stack([x1, y1, x2, y2], axis=1)
            keep = nms(boxes, scores, IOU_THRESHOLD)
            boxes = boxes[keep]
            scores = scores[keep]
        else:
            boxes = np.empty((0, 4))
            scores = np.empty((0,))
        postprocess_time = time.perf_counter() - postprocess_start

        inference_times.append((preprocess_time, inference_time, postprocess_time))

        # Print results for first frame
        if frame_count == 1:
            print(f"Preprocess time: {preprocess_time*1000:.2f} ms")
            print(f"Inference time: {inference_time*1000:.2f} ms")
            print(f"Postprocess time: {postprocess_time*1000:.2f} ms")
            print(f"Detections: {len(boxes)}")
            for i, box in enumerate(boxes):
                print(f"Box {i}: {box}, score: {scores[i]:.2f}")

    cap.release()
    print(f"Frames processed: {frame_count}")
    avg_pre = np.mean([t[0] for t in inference_times])
    avg_inf = np.mean([t[1] for t in inference_times])
    avg_post = np.mean([t[2] for t in inference_times])
    print(f"Average preprocess: {avg_pre*1000:.2f} ms")
    print(f"Average inference: {avg_inf*1000:.2f} ms")
    print(f"Average postprocess: {avg_post*1000:.2f} ms")

if __name__ == "__main__":
    main()