import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Load YOLOv12 model
model = YOLO("pumpkin_yolov12.pt")  # Replace with your model path

# Video source
cap = cv2.VideoCapture("pumpkin_field_video.mp4")  # or 0 for webcam

# Output video writer setup
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_pumpkin_with_ids.mp4", fourcc, fps_input, (frame_width, frame_height))

# Tracking state
CONFIDENCE_THRESHOLD = 0.5
next_pumpkin_id = 0
pumpkin_tracks = {}   # id -> center
active_ids = set()
total_counted_ids = set()

# FPS timer
fps = 0
prev_time = cv2.getTickCount()

def get_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def is_same_object(c1, c2, threshold=50):
    return np.linalg.norm(np.array(c1) - np.array(c2)) < threshold

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # FPS calculation
    curr_time = cv2.getTickCount()
    time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
    fps = 1.0 / time_diff if time_diff > 0 else 0
    prev_time = curr_time

    results = model(frame)[0]
    current_frame_ids = {}
    current_centers = []

    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        label = model.names[int(cls)]

        if label.lower() != "pumpkin":
            continue

        center = get_center([x1, y1, x2, y2])
        current_centers.append((box, center))

    matched_ids = {}
    used_prev_ids = set()

    for box, center in current_centers:
        found_match = False
        for pid, prev_center in pumpkin_tracks.items():
            if pid in used_prev_ids:
                continue
            if is_same_object(center, prev_center):
                matched_ids[pid] = (box, center)
                used_prev_ids.add(pid)
                found_match = True
                break
        if not found_match:
            # Assign new ID
            matched_ids[next_pumpkin_id] = (box, center)
            total_counted_ids.add(next_pumpkin_id)
            next_pumpkin_id += 1

    # Update pumpkin_tracks
    pumpkin_tracks = {pid: center for pid, (box, center) in matched_ids.items()}

    # Draw results
    for pid, (box, center) in matched_ids.items():
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Pumpkin ID: {pid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Overlay for FPS and count
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (420, 90), (255, 255, 255), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(frame, f"Total Unique Pumpkins: {len(total_counted_ids)}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Show and save frame
    cv2.imshow("Pumpkin Detection with IDs", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
out.release()
cv2.destroyAllWindows()
