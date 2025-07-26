"""
import cv2 as cv
import easyocr

reader = easyocr.Reader(["en"])

def recognize_lp(frame, lp_points):
    x1, y1, x2, y2 = lp_points
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return ""
    
    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    resized = cv.resize(gray, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    _, thresh = cv.threshold(resized, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    text = reader.readtext(thresh)
    if len(text) == 0:
        print("No text detected")
        return ""
    else:
        texts = [t[1] for t in text]
        print("Detected texts:", texts)
        return " ".join(texts)

"""

import cv2 as cv
import easyocr
from collections import defaultdict, Counter

reader = easyocr.Reader(["en"])
plate_history = defaultdict(list)  # Stores recent plate texts per car_id for smoothing

def get_stable_text(car_id, window=5):
    
    # Get the most frequent text in the recent window to stabilize output
    texts = plate_history[car_id][-window:]
    if not texts:
        return ""
    return Counter(texts).most_common(1)[0][0]

def recognize_lp(frame, lp_points, car_id=None, frame_id=None, conf_threshold=0.5):
    x1, y1, x2, y2 = lp_points
    h, w = frame.shape[:2]

    # Ensure crop coordinates stay inside the image boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return ""

    # Preprocess: grayscale, resize, and threshold for better OCR accuracy
    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    resized = cv.resize(gray, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    _, thresh = cv.threshold(resized, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Run OCR and filter out low-confidence results
    results = reader.readtext(thresh)
    filtered_texts = [text for bbox, text, conf in results if conf >= conf_threshold]

    if not filtered_texts:
        print("Low confidence or no text detected")

        # Return stable text from history if available
        return get_stable_text(car_id) if car_id is not None else ""
    
    joined_text = " ".join(filtered_texts)
    print("Detected text (filtered):", joined_text)

    if car_id is not None:
       
        # Update text history every 3rd frame to reduce noise
        if frame_id is None or frame_id % 3 == 0:
            plate_history[car_id].append(joined_text)

        # Return the stabilized text from recent history
        return get_stable_text(car_id)
    
    return joined_text



