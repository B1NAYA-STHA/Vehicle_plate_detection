import cv2 as cv

def draw_label(frame, text, x, y, color=(0, 255, 0)):
    font = cv.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    
    # Get text size for background rectangle
    (w, h), _ = cv.getTextSize(text, font, scale, thickness)
    
    # Draw filled rectangle as label background slightly above the point (x, y)
    cv.rectangle(frame, (x, y - h - 10), (x + w + 10, y), color, -1)
    
    # Put text on top of the rectangle with black color for good contrast
    cv.putText(frame, text, (x + 5, y - 5), font, scale, (0, 0, 0), thickness, cv.LINE_AA)
