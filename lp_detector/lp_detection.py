import cv2 as cv
from ultralytics import YOLO
from .lp_recognizer import recognize_lp
from draw_label import draw_label

lp_model = YOLO("Vehicle_plate_detection\model\\best.pt") #license plate detection model

def detect_lp(frame, vehicle_box):
    x1, y1, x2, y2 = vehicle_box
    #Crop the car from each frame if the video
    vehicle_crops = frame[y1:y2, x1:x2]

    # Run license plate detection model on the cropped vehicle region
    lp_results = lp_model(vehicle_crops)[0]

    for box in lp_results.boxes:
        lp_x1, lp_y1, lp_x2, lp_y2 = map(int, box.xyxy[0])
        
        # Adjust coordinates relative to original frame
        off_x1, off_y1 = lp_x1 + x1, lp_y1 + y1
        off_x2, off_y2 = lp_x2 + x1, lp_y2 + y1
        
        # Recognize license plate text using EasyOCR
        lp_text = recognize_lp(frame, (off_x1, off_y1, off_x2, off_y2))
        
        # Draw bounding box and recognized text
        cv.rectangle(frame, (off_x1, off_y1), (off_x2, off_y2), (0, 0, 255), 3, cv.LINE_AA)
        draw_label(frame, lp_text, off_x1, off_y1, (0, 0, 255))

    return frame
