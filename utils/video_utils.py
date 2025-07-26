import cv2 as cv
from .classes import classNames
import supervision as sv
import numpy as np
from lp_detector import detect_lp
from draw_label import draw_label

class_id = [2, 3, 5, 7]  # IDs for car, motorbike, bus, truck
byte_tracker = sv.ByteTrack()

# def draw_boxes(frame, result):
#     for r in result:
#         boxes = r.boxes
#         for box in boxes:
#             #bounding boxes coords
#             x1, y1, x2, y2 = map(int, box.xyxy[0])

#             #rounding up confidence
#             conf = math.ceil((box.conf[0] * 100)) / 100

#             #classlabel of the detected object
#             cls = int(box.cls[0])

#             #label for detected object
#             label = f"{classNames[cls]}: {conf}"

#             if classNames[cls] in ["car", "motorbike", "bus", "truck"]:
#                 cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3, cv.LINE_AA)
#                 draw_label(frame, label, x1, y1, (255, 0, 255))
#     return frame   


def draw_boxes(frame, result, model, frame_count, skip_rate=3):
    # Filter detections for selected vehicle classes
    detections = sv.Detections.from_ultralytics(result[0]) 
    detections = detections[np.isin(detections.class_id, class_id)]

    # Track detected objects
    tracker = byte_tracker.update_with_detections(detections)

    for detection in tracker:
        if detection[4] == -1:
            continue

        x1, y1, x2, y2 = detection[0].astype(int)
        track_id = detection[4]
        class_name = model.names[detection[3]]
        label = f"{track_id}: {class_name}"

        # Draw bounding box and label on frame
        cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3, cv.LINE_AA)
        draw_label(frame, label, x1, y1, (255, 0, 255))

        # Run license plate detection inside vehicle bounding box
        frame = detect_lp(frame, (x1, y1, x2, y2))

    return frame
