import cv2 as cv

def video_writer(output_video_url, fps, frame_size):

    # Initialize VideoWriter with MP4 codec, fps, and frame size
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    return cv.VideoWriter(output_video_url, fourcc, fps, frame_size)

def save_video(video_write, frame):
    
    # Write a single frame to the video file
    video_write.write(frame)
