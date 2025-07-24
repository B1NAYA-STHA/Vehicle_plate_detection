import cv2 as cv

def video_writer(output_video_url, fps, frame_size):
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    return cv.VideoWriter(output_video_url, fourcc, fps, frame_size)

def save_video(video_write, frame):
    video_write.write(frame)