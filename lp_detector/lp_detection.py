import cv2 as cv
from ultralytics import YOLO

lp_model = YOLO("Vehicle_plate_detection\model\\best.pt")

def detect_lp(frame, vehicle_box):
    x1, y1, x2, y2 = vehicle_box
    vehicle_crops = frame[y1:y2, x1:x2]

    lp_results = lp_model(vehicle_crops)[0]

    for box in lp_results.boxes:
        lp_x1, lp_y1, lp_x2, lp_y2 = map(int, box.xyxy[0])
        off_x1, off_y1 = lp_x1 + x1, lp_y1 + y1
        off_x2, off_y2 = lp_x2 + x1, lp_y2 + y1
        cv.rectangle(frame, (off_x1, off_y1), (off_x2, off_y2), (0, 0, 255), 3, cv.LINE_AA)

    return frame