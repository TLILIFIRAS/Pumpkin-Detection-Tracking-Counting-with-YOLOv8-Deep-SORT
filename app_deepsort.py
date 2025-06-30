import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# === CONFIGURATION ===
YOLO_MODEL_PATH = "pumpkin_yolov12.pt"
VIDEO_SOURCE = "pumpkin_field_video.mp4"  # or 0 for webcam
OUTPUT_VIDEO_PATH = "output_pumpkin_deepsort_optimized.mp4"
CONFIDENCE_THRESHOLD = 0.5
DRAW_BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 255)

# === INITIALIZE ===
model = YOLO(YOLO_MODEL_PATH)
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(VIDEO_SOURCE)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS) or 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps_input, (frame_width, frame_height))

# === TRACKING STATE ===
prev_time = cv2.getTickCount()
counted_ids = set()

# === MAIN LOOP ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # === FPS Calculation ===
    curr_time = cv2.getTickCount()
    time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
    fps = 1.0 / time_diff if time_diff > 0 else 0
    prev_time = curr_time

    # === YOLOv12 Detection ===
    results = model(frame)[0]
    detections = []

    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        if conf < CONFIDENCE_THRESHOLD:
            continue

        class_name = model.names[int(cls)]
        if class_name.lower() != "pumpkin":
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        w = x2 - x1
        h = y2 - y1
        if w < 15 or h < 15:
            continue

        detections.append(([x1, y1, w, h], conf.item(), "pumpkin"))

    # === Deep SORT Tracking ===
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        counted_ids.add(track_id)

        # === Draw bounding box and ID ===
        cv2.rectangle(frame, (l, t), (r, b), DRAW_BOX_COLOR, 2)
        cv2.putText(frame, f"Pumpkin ID: {track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)

    # === White Overlay for FPS and Count ===
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (420, 90), (255, 255, 255), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(frame, f"Total Unique Pumpkins: {len(counted_ids)}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # === Display and Save ===
    cv2.imshow("Pumpkin Detection with Deep SORT", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
out.release()
cv2.destroyAllWindows()
