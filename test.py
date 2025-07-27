import cv2 as cv
import numpy as np
import pandas as pd
from ultralytics import YOLO
import easyocr
import os

model = YOLO("Vehicle_plate_detection/model/best.pt")
img = cv.imread("Vehicle_plate_detection/photos/5.jpg")

reader = easyocr.Reader(['en'])
csv_file = "Vehicle_plate_detection\csv_files\info.csv"

if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=["plate_bbox", "plate_num", "conf"])
    df.to_csv(csv_file, index=False)
# Run YOLO prediction

results = model(img)[0]

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        text = reader.readtext(crop)
        new_data = pd.DataFrame([{
            "plate_bbox": (x1, y1, x2, y2),
            "plate_num": text[0][1],
            "conf": text[0][2]
        }])
        new_data.to_csv(csv_file, mode='a', header=False, index=False)
        print(f"Detected Text: {text[0][1]}")

# Show the final image
cv.imshow("Plate Detection", img)
cv.waitKey(0)
cv.destroyAllWindows()
