from ultralytics import YOLO
import cv2 as cv
import supervision as sv
import numpy as np
import easyocr

model = YOLO("Vehicle_plate_detection\model\\best.pt")
img = cv.imread("Vehicle_plate_detection\photos\\1.jpg")


result = model(img)
reader = easyocr.Reader(['en'])

for r in result:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]

        ocr_results = reader.readtext(crop)
        for (_, text, conf) in ocr_results:
            print(f"Detected text: {text}, Confidence: {conf}")

        # Draw result on image
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if ocr_results:
            cv.putText(img, ocr_results[0][1], (x1, y1 - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
cv.imshow('car', img)
cv.waitKey(0)
    
