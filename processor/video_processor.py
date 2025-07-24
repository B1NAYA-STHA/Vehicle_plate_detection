import cv2 as cv

def get_frame_and_result(model, video_url):
    cap = cv.VideoCapture(video_url)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        frame = cv.resize(img, (1280, 720))
        results = model(frame)
        yield frame, results
    
    cap.release
