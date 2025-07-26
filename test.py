import cv2 as cv
import numpy as np
from ultralytics import YOLO
import easyocr

model = YOLO("Vehicle_plate_detection/model/best.pt")
img = cv.imread("Vehicle_plate_detection/photos/5.jpg")

reader = easyocr.Reader(['en'])

# Run YOLO prediction
results = model(img)[0]

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        text = reader.readtext(crop)
        print(len(text))
        print(f"Detected Text: {text[0][1]}")

# Show the final image
cv.imshow("Plate Detection", img)
cv.waitKey(0)
cv.destroyAllWindows()
