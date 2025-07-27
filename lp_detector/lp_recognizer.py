import cv2 as cv
import easyocr
from collections import defaultdict, Counter
from csv_files import update_csv

# Initialize EasyOCR reader for English language
reader = easyocr.Reader(["en"])

# Dictionary to store history of detected license plates per car track_id
plate_history = defaultdict(list)

def get_stable_text(car_id, window=5):
    """
    Returns the most frequently occurring license plate text in the latest `window` entries
    for a given car_id. Helps smooth out noisy OCR results.
    """
    texts = plate_history[car_id][-window:]
    return Counter(texts).most_common(1)[0][0] if texts else ""


def recognize_lp(frame, lp_points, frame_count, track_id, conf_threshold=0.5):

    x1, y1, x2, y2 = lp_points
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "", 0.0

    # --- Preprocessing for better OCR ---
    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    resized = cv.resize(gray, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    _, thresh = cv.threshold(resized, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    results = reader.readtext(thresh)

    # Filter results by confidence threshold
    filtered = [(text, conf) for _, text, conf in results if conf >= conf_threshold]

    if not filtered:
        stable = get_stable_text(track_id)
        return stable, 0.0

    texts = [t for t, _ in filtered]
    joined = " ".join(texts)

    # Choose the maximum confidence score from the filtered results
    lp_conf = max(conf for _, conf in filtered)

    update_csv(
        frame_number=frame_count,
        track_id=track_id,
        lp_text=joined,
        lp_conf=lp_conf
    )

    # Add the detected plate to the history every 3rd frame to smooth predictions
    if frame_count % 3 == 0:
        plate_history[track_id].append(joined)

    # Return the most frequent (smoothed) text from history and the confidence
    return get_stable_text(track_id), lp_conf
