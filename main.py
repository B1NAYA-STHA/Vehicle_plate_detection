from ultralytics import YOLO
import cv2 as cv
from utils import draw_boxes
from processor import get_frame_and_result

model = YOLO("yolov8n.pt")
video_url = "Vehicle_license_plate_detection\\videos\\3.mp4"

for frame, results in get_frame_and_result(model, video_url):
    annotated_frame = draw_boxes(frame, results, model)
    cv.imshow("car", annotated_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows