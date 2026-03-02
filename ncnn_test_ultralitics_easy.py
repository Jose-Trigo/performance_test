import cv2
import time
import numpy as np
from ultralytics import YOLO
from queue import Queue
import threading
from collections import defaultdict
import psutil
import os

# Configuration
video_path = "brt_presentation.mp4"
model_path = "yolo_custom_v2_mapilary_ncnn_model"  ## for ncnn its a folder with the .params and .bin files inside
YOLO_CONFIDENCE = 0.4
CROP_PADDING = 5
TRACKER_MAX_AGE = 50
TRACKER_MIN_HITS = 1
TRACKER_IOU_THRESHOLD = 0.3
DISPLAY_SCALE = 0.5
SHOW_LIVE = True

# Queues
frame_queue = Queue(maxsize=2)
crop_queue = Queue(maxsize=200)
visualization_queue = Queue(maxsize=10)  # NEW: separate queue for visualization
stop_event = threading.Event()

# Timing statistics
timing_stats = defaultdict(list)
timing_lock = threading.Lock()

# Track IDs for SORT
from filterpy.kalman import KalmanFilter

# Track colors for visualization
np.random.seed(42)
TRACK_COLORS = {}
def get_track_color(track_id):
    if track_id not in TRACK_COLORS:
        TRACK_COLORS[track_id] = tuple(map(int, np.random.randint(50, 255, 3)))
    return TRACK_COLORS[track_id]

# ============================================
# SORT TRACKER (same as before)
# ============================================
def linear_assignment(cost_matrix):
    try:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))
    except ImportError:
        matches = []
        cost = cost_matrix.copy()
        while cost.size > 0 and cost.shape[0] > 0 and cost.shape[1] > 0:
            min_idx = np.argmin(cost)
            i, j = np.unravel_index(min_idx, cost.shape)
            matches.append([i, j])
            cost = np.delete(cost, i, axis=0)
            cost = np.delete(cost, j, axis=1)
        return np.array(matches) if matches else np.empty((0, 2), dtype=int)

def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return(o)

def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h
    r = w / float(h) if h != 0 else 1.0
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w if w != 0 else 1.0
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

class KalmanBoxTracker(object):
    count = 0
    
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    iou_matrix = iou_batch(detections, trackers)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))
    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
    def __init__(self, max_age=30, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5)), frame_id=None):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,5))

# ============================================
# CAMERA THREAD
# ============================================
def camera_thread(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        stop_event.set()
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / fps if fps > 0 else 0.033
    frame_count = 0
    dropped_frames = 0
    
    print(f"[CAMERA] Started (FPS: {fps:.2f})")
    start_time = time.time()
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[CAMERA] End of video")
            stop_event.set()
            break
        
        frame_count += 1
        current_timestamp = time.time()
        
        # Clear queue and keep only most recent frame
        try:
            while not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                    dropped_frames += 1
                except:
                    break
            
            frame_queue.put({
                'frame_id': frame_count,
                'frame': frame.copy(),
                'timestamp': current_timestamp
            }, block=False)
        except:
            dropped_frames += 1
        
        # Maintain timing
        expected_time = start_time + (frame_count * frame_delay)
        sleep_time = expected_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    cap.release()
    print(f"[CAMERA] Stopped after {frame_count} frames (dropped {dropped_frames})")

# ============================================
# YOLO THREAD - With visualization data (NOT TIMED)
# ============================================
def yolo_thread(model, mot_tracker):
    print("[YOLO] Started with detailed timing + visualization")
    processed_frames = 0
    total_crops_created = 0
    
    while not stop_event.is_set():
        try:
            frame_data = frame_queue.get(timeout=0.1)
        except:
            continue
        
        frame_count = frame_data['frame_id']
        frame = frame_data['frame']
        current_timestamp = frame_data['timestamp']
        
        # ========================================
        # ALL TIMING STARTS HERE (FRAME ACQUISITION FROM QUEUE IS INCLUDED)
        # ========================================
        frame_start = time.perf_counter()
        
        # ===== STEP 1: Frame preprocessing =====
        preprocess_start = time.perf_counter()
        height, width = frame.shape[:2]
        preprocess_time = time.perf_counter() - preprocess_start
        
        # ===== STEP 2: YOLO inference =====
        inference_start = time.perf_counter()
        results = model(frame, verbose=False, conf=YOLO_CONFIDENCE)
        inference_time = time.perf_counter() - inference_start
        
        # ===== STEP 3: Extract boxes from results =====
        boxes_start = time.perf_counter()
        boxes = results[0].boxes
        original_detections = []
        sort_detections = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf)
            
            # Add padding
            padding = CROP_PADDING
            x1_padded = max(0, x1 - padding)
            y1_padded = max(0, y1 - padding)
            x2_padded = min(width, x2 + padding)
            y2_padded = min(height, y2 + padding)
            
            original_detections.append({
                'bbox_raw': (x1, y1, x2, y2),
                'bbox_padded': (x1_padded, y1_padded, x2_padded, y2_padded),
                'conf': conf,
                'used': False
            })
            
            sort_detections.append([float(x1), float(y1), float(x2), float(y2), conf])
        
        boxes_time = time.perf_counter() - boxes_start
        
        # ===== STEP 4: SORT tracking =====
        tracking_start = time.perf_counter()
        dets = np.array(sort_detections) if sort_detections else np.empty((0, 5))
        tracked_objects = mot_tracker.update(dets, frame_id=frame_count)
        tracking_time = time.perf_counter() - tracking_start
        
        # ===== STEP 5: Match tracked objects to original detections =====
        matching_start = time.perf_counter()
        
        matched_tracks = []  # Store for visualization
        
        for track in tracked_objects:
            sort_x1, sort_y1, sort_x2, sort_y2, track_id = track
            track_id = int(track_id)
            
            # Find best matching original detection
            best_match_idx = -1
            best_iou = 0
            sort_box = np.array([[sort_x1, sort_y1, sort_x2, sort_y2]])
            
            for idx, orig in enumerate(original_detections):
                if orig['used']:
                    continue
                ox1, oy1, ox2, oy2 = orig['bbox_raw']
                orig_box = np.array([[float(ox1), float(oy1), float(ox2), float(oy2)]])
                iou = iou_batch(sort_box, orig_box)[0, 0]
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = idx
            
            if best_match_idx < 0 or best_iou < 0.1:
                continue
            
            original_detections[best_match_idx]['used'] = True
            x1, y1, x2, y2 = original_detections[best_match_idx]['bbox_padded']
            
            matched_tracks.append({
                'track_id': track_id,
                'bbox': (x1, y1, x2, y2),
                'conf': original_detections[best_match_idx]['conf']
            })
        
        matching_time = time.perf_counter() - matching_start
        
        # ===== STEP 6: Create crops =====
        cropping_start = time.perf_counter()
        crops_created = 0
        
        for match in matched_tracks:
            track_id = match['track_id']
            x1, y1, x2, y2 = match['bbox']
            
            # Actual cropping operation
            crop = frame[y1:y2, x1:x2]
            crop = np.ascontiguousarray(crop)
            
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
            
            # Queue the crop
            crop_data = {
                'frame_id': frame_count,
                'timestamp': current_timestamp,
                'object_id': track_id,
                'crop': crop.copy(),
                'bbox': (x1, y1, x2, y2)
            }
            
            try:
                crop_queue.put(crop_data, block=False)
                crops_created += 1
            except:
                pass
        
        cropping_time = time.perf_counter() - cropping_start
        
        # ========================================
        # TIMING ENDS HERE (BEFORE VISUALIZATION)
        # ========================================
        frame_total_time = time.perf_counter() - frame_start
        
        # Store timing statistics
        with timing_lock:
            timing_stats['frame_total'].append(frame_total_time)
            timing_stats['preprocessing'].append(preprocess_time)
            timing_stats['yolo_inference'].append(inference_time)
            timing_stats['boxes_extraction'].append(boxes_time)
            timing_stats['sort_tracking'].append(tracking_time)
            timing_stats['detection_matching'].append(matching_time)
            timing_stats['cropping'].append(cropping_time)
            timing_stats['detections_per_frame'].append(len(original_detections))
            timing_stats['tracks_per_frame'].append(len(tracked_objects))
            timing_stats['crops_per_frame'].append(crops_created)
        
        processed_frames += 1
        total_crops_created += crops_created
        
        # ========================================
        # VISUALIZATION (NOT TIMED - AFTER ALL PROCESSING)
        # ========================================
        viz_data = {
            'frame_id': frame_count,
            'frame': frame.copy(),
            'tracked_objects': matched_tracks,
            'total_detections': len(original_detections),
            'total_tracks': len(tracked_objects),
            'crops_created': crops_created,
            'frame_time_ms': frame_total_time * 1000
        }
        
        try:
            visualization_queue.put(viz_data, block=False)
        except:
            pass  # Drop visualization if queue is full
        
        # Print progress every 100 frames
        if processed_frames % 100 == 0:
            print(f"[YOLO] Processed {processed_frames} frames | "
                  f"Avg inference: {np.mean(timing_stats['yolo_inference'])*1000:.2f}ms | "
                  f"Total crops: {total_crops_created}")
    
    print(f"[YOLO] Stopped after processing {processed_frames} frames")

# ============================================
# VISUALIZATION THREAD (SEPARATE, NOT AFFECTING TIMING)
# ============================================
def visualization_thread(width, height):
    print("[VISUALIZATION] Started")
    
    display_width = int(width * DISPLAY_SCALE)
    display_height = int(height * DISPLAY_SCALE)
    
    cv2.namedWindow("YOLO Benchmark - Live", cv2.WINDOW_NORMAL)
    
    while not stop_event.is_set():
        try:
            viz_data = visualization_queue.get(timeout=0.1)
        except:
            continue
        
        frame_id = viz_data['frame_id']
        frame = viz_data['frame']
        tracked_objects = viz_data['tracked_objects']
        total_detections = viz_data['total_detections']
        total_tracks = viz_data['total_tracks']
        crops_created = viz_data['crops_created']
        frame_time_ms = viz_data['frame_time_ms']
        
        # Create annotated frame
        annotated = frame.copy()
        
        # Draw tracked objects
        for obj in tracked_objects:
            track_id = obj['track_id']
            x1, y1, x2, y2 = obj['bbox']
            conf = obj['conf']
            
            color = get_track_color(track_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            label_y = y1 - 10 if y1 > 30 else y2 + 20
            cv2.putText(annotated, f"ID:{track_id} | Conf:{conf:.2f}",
                       (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add header info
        cv2.putText(annotated, f"Frame: {frame_id} | YOLO Benchmark", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add stats overlay (bottom)
        overlay_height = 120
        cv2.rectangle(annotated, (10, height - overlay_height - 10), (600, height - 10), (0, 0, 0), -1)
        
        y_offset = height - overlay_height
        cv2.putText(annotated, f"Frame Processing Time: {frame_time_ms:.2f} ms", 
                   (20, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(annotated, f"Detections: {total_detections} | Tracked: {total_tracks} | Crops: {crops_created}",
                   (20, y_offset + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        with timing_lock:
            if len(timing_stats['yolo_inference']) > 0:
                avg_inference = np.mean(timing_stats['yolo_inference']) * 1000
                avg_total = np.mean(timing_stats['frame_total']) * 1000
                cv2.putText(annotated, f"Avg YOLO Inference: {avg_inference:.2f} ms | Avg Total: {avg_total:.2f} ms",
                           (20, y_offset + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                total_crops = sum(timing_stats['crops_per_frame'])
                cv2.putText(annotated, f"Total Crops Generated: {total_crops}",
                           (20, y_offset + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Display
        display_frame = cv2.resize(annotated, (display_width, display_height))
        cv2.imshow("YOLO Benchmark - Live", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[VISUALIZATION] User pressed 'q' - Stopping...")
            stop_event.set()
            break
    
    cv2.destroyAllWindows()
    print("[VISUALIZATION] Stopped")

# ============================================
# MAIN
# ============================================
def main():
    print("=" * 80)
    print("YOLO PIPELINE BENCHMARK - With Visualization (Not affecting timing)")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    print(f"YOLO Confidence: {YOLO_CONFIDENCE}")
    print(f"Crop Padding: {CROP_PADDING}px")
    print(f"Tracker: SORT (max_age={TRACKER_MAX_AGE}, min_hits={TRACKER_MIN_HITS}, iou={TRACKER_IOU_THRESHOLD})")
    print("=" * 80)
    print("\nTIMING EXPLANATION:")
    print("  • Frame acquisition from queue: INCLUDED in frame_total")
    print("  • Preprocessing: INCLUDED")
    print("  • YOLO inference: INCLUDED")
    print("  • Boxes extraction: INCLUDED")
    print("  • SORT tracking: INCLUDED")
    print("  • Detection matching: INCLUDED")
    print("  • Cropping & queue insertion: INCLUDED")
    print("  • Visualization: NOT INCLUDED (runs in separate thread)")
    print("=" * 80)
    
    # Load YOLO model
    print("\n[SETUP] Loading YOLO model...")
    model = YOLO(model_path)
    
    # Get video info
    cap_temp = cv2.VideoCapture(video_path)
    width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_temp.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_temp.release()
    
    print(f"[SETUP] Video: {width}x{height} @ {fps:.2f} FPS ({total_frames} frames)")
    
    # Initialize tracker
    mot_tracker = Sort(max_age=TRACKER_MAX_AGE, 
                      min_hits=TRACKER_MIN_HITS, 
                      iou_threshold=TRACKER_IOU_THRESHOLD)
    
    # Start threads
    print("\n[SETUP] Starting threads...")
    threads = [
        threading.Thread(target=camera_thread, args=(video_path,), name="Camera", daemon=True),
        threading.Thread(target=yolo_thread, args=(model, mot_tracker), name="YOLO", daemon=True),
        threading.Thread(target=visualization_thread, args=(width, height), name="Visualization", daemon=True) if SHOW_LIVE else None
    ]
    
    threads = [t for t in threads if t is not None]
    
    for t in threads:
        t.start()
    
    print("[BENCHMARK] Running... Press 'q' in window to quit\n")
    
    # Wait for completion
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n[BENCHMARK] Interrupted by user")
        stop_event.set()
        for t in threads:
            t.join(timeout=2)
    
    # ============================================
    # RESULTS
    # ============================================
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    with timing_lock:
        if len(timing_stats['frame_total']) == 0:
            print("No frames processed!")
            return
        
        processed_count = len(timing_stats['frame_total'])
        
        print(f"\n📊 FRAMES PROCESSED: {processed_count}")
        print(f"📊 TOTAL DETECTIONS: {sum(timing_stats['detections_per_frame'])}")
        print(f"📊 TOTAL TRACKED OBJECTS: {sum(timing_stats['tracks_per_frame'])}")
        print(f"📊 TOTAL CROPS CREATED: {sum(timing_stats['crops_per_frame'])}")
        
        print("\n" + "-" * 80)
        print("TIMING BREAKDOWN (milliseconds)")
        print("-" * 80)
        
        categories = [
            ('TOTAL FRAME PROCESSING', 'frame_total'),
            ('  ├─ Preprocessing', 'preprocessing'),
            ('  ├─ YOLO Inference', 'yolo_inference'),
            ('  ├─ Boxes Extraction', 'boxes_extraction'),
            ('  ├─ SORT Tracking', 'sort_tracking'),
            ('  ├─ Detection Matching', 'detection_matching'),
            ('  └─ Cropping', 'cropping'),
        ]
        
        for label, key in categories:
            data = timing_stats[key]
            avg = np.mean(data) * 1000
            std = np.std(data) * 1000
            min_val = np.min(data) * 1000
            max_val = np.max(data) * 1000
            p50 = np.percentile(data, 50) * 1000
            p95 = np.percentile(data, 95) * 1000
            p99 = np.percentile(data, 99) * 1000
            
            print(f"\n{label}:")
            print(f"  Average: {avg:7.2f} ms  ±{std:6.2f} ms")
            print(f"  Min/Max: {min_val:7.2f} ms / {max_val:7.2f} ms")
            print(f"  P50/P95/P99: {p50:7.2f} ms / {p95:7.2f} ms / {p99:7.2f} ms")
        
        print("\n" + "-" * 80)
        print("THROUGHPUT")
        print("-" * 80)
        
        avg_total_time = np.mean(timing_stats['frame_total'])
        theoretical_fps = 1.0 / avg_total_time if avg_total_time > 0 else 0
        
        print(f"Average time per frame: {avg_total_time*1000:.2f} ms")
        print(f"Theoretical FPS: {theoretical_fps:.2f}")
        print(f"Processing speed: {theoretical_fps/fps*100:.1f}% of real-time" if fps > 0 else "N/A")
        
        print("\n" + "-" * 80)
        print("DETECTIONS & TRACKING")
        print("-" * 80)
        
        avg_detections = np.mean(timing_stats['detections_per_frame'])
        avg_tracks = np.mean(timing_stats['tracks_per_frame'])
        avg_crops = np.mean(timing_stats['crops_per_frame'])
        
        print(f"Average detections per frame: {avg_detections:.2f}")
        print(f"Average tracked objects per frame: {avg_tracks:.2f}")
        print(f"Average crops created per frame: {avg_crops:.2f}")
        
        print("\n" + "=" * 80)
        print("SYSTEM INFO")
        print("=" * 80)
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        print(f"CPU Usage: {cpu_percent:.1f}%")
        print(f"Memory Usage: {memory.percent:.1f}% ({memory.used / (1024**3):.1f} GB / {memory.total / (1024**3):.1f} GB)")
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    main()