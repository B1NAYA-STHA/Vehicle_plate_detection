import cv2 as cv

def get_frame_and_result(model, video_url):
    cap = cv.VideoCapture(video_url)  # Open video file or stream

    while True:
        ret, img = cap.read()  # Read a frame
        if not ret:  
            break

        frame = cv.resize(img, (1280, 720)) 
        results = model(frame)  # Run model inference on the frame

        yield frame, results  # Yield frame and model results as a generator

    cap.release() 
