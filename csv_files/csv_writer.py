import pandas as pd
import os

csv_file = "Vehicle_plate_detection\csv_files\combined_output.csv"

# create columns for the csv file
def init_csv():
    if not os.path.exists(csv_file):
        df = pd.DataFrame(columns=[
            "frame_number", "track_id", "car_bbox", "lp_bbox", "lp_text", "lp_conf"
        ])
        df.to_csv(csv_file, index=True)

# return bbox values as string
def clean_bbox(bbox):
    if bbox:
        return str(tuple(int(x) for x in bbox))
    return None

# Update or append a row
def update_csv(frame_number, track_id=None, car_bbox=None, lp_bbox=None, lp_text=None, lp_conf=None):

    if not lp_text:
        return
    
    init_csv()
    df = pd.read_csv(csv_file, index_col=0)


    row_index = (df["frame_number"] == frame_number) & ((df["track_id"] == track_id) if track_id is not None else True)
    matched = df[row_index]

    if matched.empty:
        # Append new row
        new_row = {
            "frame_number": frame_number,
            "track_id": track_id,
            "car_bbox": clean_bbox(car_bbox),
            "lp_bbox": clean_bbox(lp_bbox),
            "lp_text": lp_text,
            "lp_conf": lp_conf
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        # Update existing row
        idx = matched.index[0]
        if car_bbox is not None:
            df.at[idx, "car_bbox"] = clean_bbox(car_bbox)
        if lp_bbox is not None:
            df.at[idx, "lp_bbox"] = clean_bbox(lp_bbox)
        if lp_text is not None:
            df.at[idx, "lp_text"] = lp_text
        if lp_conf is not None:
            df.at[idx, "lp_conf"] = lp_conf

    df.to_csv(csv_file, index=True)


