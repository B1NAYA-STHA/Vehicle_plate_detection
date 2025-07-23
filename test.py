from ultralytics import YOLO
import cv2 as cv
import math
import supervision as sv
import numpy as np

def draw_label(frame, text, x, y, color=(0, 255, 0)):
    font = cv.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (w, h), _ = cv.getTextSize(text, font, scale, thickness)
    cv.rectangle(frame, (x, y - h - 10), (x + w + 10, y), color, -1)  # Label background
    cv.putText(frame, text, (x + 5, y - 5), font, scale, (0, 0, 0), thickness, cv.LINE_AA)  # Black text for contrast

model = YOLO("yolov8n.pt")
# img = cv.imread("Vehicle_license_plate_detection\photos\\2.jpg")
# result = model(img)
# result[0].show()

cap = cv.VideoCapture("Vehicle_license_plate_detection\\videos\\1.mp4")
class_id = [2, 3, 5, 7]
byte_tracker = sv.ByteTrack()
while True:
    ret, img = cap.read()
    frame = cv.resize(img, (1280, 720))
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result) 
    detections = detections[np.isin(detections.class_id, class_id)]

    tracker = byte_tracker.update_with_detections(detections)
    for detection in tracker:
        if detection[4] != -1:
            x1, y1, x2, y2 = detection[0].astype(int)
            track_id = detection[4]
            class_name = model.names[detection[3]]
            label = f"{track_id}: {class_name}"
            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3, cv.LINE_AA)
            draw_label(frame, label, x1, y1, (255, 0, 255))
        
    cv.imshow("car", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release
cv.destroyAllWindows