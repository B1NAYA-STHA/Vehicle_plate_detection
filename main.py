from ultralytics import YOLO
import cv2 as cv
from utils import draw_boxes, save_video, video_writer
from processor import get_frame_and_result

model = YOLO("yolov8n.pt") #pre--trained YOLO model
video_url = "Vehicle_plate_detection\\videos\\5.mp4"
output_video_url = "Vehicle_plate_detection\\output\\2.mp4"
frame_count = 0

# Initialize video writer with output path, fps, and frame size
video_write = video_writer(output_video_url, fps=30, frame_size=(1280, 720))

for frame, results in get_frame_and_result(model, video_url):
    frame_count += 1

    # Draw bounding boxes and annotations on the frame
    annotated_frame = draw_boxes(frame, results, model, frame_count)

    # Save the annotated frame to the output video
    save_video(video_write, annotated_frame)

    # Uncomment below lines to show the video in a window
    # cv.imshow("car", annotated_frame)
    # if cv.waitKey(1) & 0xFF == ord('q'):
    #     break

video_write.release()
cv.destroyAllWindows()
