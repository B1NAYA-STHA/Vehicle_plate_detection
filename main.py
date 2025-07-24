from ultralytics import YOLO
import cv2 as cv
from utils import draw_boxes, save_video, video_writer
from processor import get_frame_and_result

model = YOLO("yolov8n.pt")
video_url = "Vehicle_plate_detection\\videos\\1.mp4"
output_video_url = "Vehicle_plate_detection\output\\3.mp4"
frame_count = 0

video_write = video_writer(output_video_url, fps=30, frame_size = (1280, 720))

for frame, results in get_frame_and_result(model, video_url):
    frame_count +=1
    annotated_frame = draw_boxes(frame, results, model, frame_count)
    save_video(video_write, annotated_frame)
    # cv.imshow("car", annotated_frame)
    # if cv.waitkey(1) & 0xFF == ord('q'):
    #     break

video_write.release
cv.destroyAllWindows