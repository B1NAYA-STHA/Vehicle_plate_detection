import cv2 as cv
from .classes import classNames
import supervision as sv
import numpy as np
from lp_detector import detect_lp
from draw_label import draw_label
from speed_estimation import speed_estimate
from collections import defaultdict, deque

class_id = [2, 3, 5, 7]  # IDs for car, motorbike, bus, truck
byte_tracker = sv.ByteTrack()

SOURCE = np.array([[417, 262],[766, 267],[1679, 719],[-183, 719]])

#create a polyzonezone where vehicle is detected within that area
polygonzone = sv.PolygonZone(polygon=SOURCE)

#coordinates = defaultdict(lambda: deque(maxlen=15))
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

def draw_boxes(frame, result, model, frame_count, fps):
    # Filter detections for selected vehicle classes
    detections = sv.Detections.from_ultralytics(result[0]) 
    detections = detections[np.isin(detections.class_id, class_id)]
    detections = detections[polygonzone.trigger(detections)]

    # Track detected objects
    tracker = byte_tracker.update_with_detections(detections)
    sv.draw_polygon(frame, polygonzone.polygon, color=sv.Color.RED, thickness=2)

    for detection in tracker:
        if detection[4] == -1:
            continue

        x1, y1, x2, y2 = detection[0].astype(int)
        track_id = detection[4]
        class_name = model.names[detection[3]]

        #get bottom center point of each vehicle
        cx = int((x1 + x2) / 2)
        cy = int(y2)

        #get speed of each vehicle
        speed = speed_estimate(np.array([[cx, cy]]), track_id, fps)
        
        label = f"{track_id}: {class_name}:  {speed}"

        # Draw bounding box and label on frame
        cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3, cv.LINE_AA)
        draw_label(frame, label, x1, y1, (255, 0, 255))

        # Run license plate detection inside vehicle bounding box
        frame = detect_lp(frame, (x1, y1, x2, y2), frame_count, track_id)

    return frame
