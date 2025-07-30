import cv2 as cv
import numpy as np
from collections import defaultdict, deque

# Perspective transformation source points in 1280x720 space
SOURCE = np.array([[417, 262], [766, 267], [1679, 719], [-183, 719]])

# Real-world target dimensions (e.g., a lane of 25m length and 250m depth)
TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

arr1 = np.float32(SOURCE)
arr2 = np.float32(TARGET)

# Generate the perspective transformation matrix
matrix = cv.getPerspectiveTransform(arr1, arr2)

# Dictionary to store the last 30 Y positions for each track ID
coordinates = defaultdict(lambda: deque(maxlen=30))

# Function to apply perspective transform to a set of points
def transform_points(points: np.ndarray, matrix) -> np.ndarray:
    if points.size == 0:
        return points

    reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
    transformed_points = cv.perspectiveTransform(reshaped_points, matrix)
    return transformed_points.reshape(-1, 2)

# Function to estimate speed using vertical displacement over time
def speed_estimate(points, track_id, fps=30):
    transformed_points = transform_points(points, matrix).astype(int)
    new_points = transformed_points.flatten()
    x, y = int(new_points[0]), int(new_points[1])
    coordinates[track_id].append(y)

    if len(coordinates[track_id]) < 15:
        return "Calculating..."
    
    y_start = coordinates[track_id][0]
    y_end = coordinates[track_id][-1]

    distance_m = abs(y_end - y_start)
    time_s = len(coordinates[track_id]) / fps
    speed_kmph = (distance_m / time_s) * 3.6

    return f"{int(speed_kmph)} km/h"
