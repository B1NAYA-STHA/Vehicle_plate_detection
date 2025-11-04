# 🚗 Vehicle Detection and License Plate Recognition using YOLOv8

This project focuses on detecting vehicles and recognizing their license plates from images and videos using the **YOLOv8** object detection model. It performs a two-stage detection process: first identifying vehicles, and then detecting license plates within those vehicles. The results are logged and saved for analysis.

---

## ⚙️ Features

- 🚙 Detects multiple vehicle types (car, truck, bus, motorbike, etc.)
- 🔍 Extracts and localizes license plates using a second-stage YOLOv8 model
- 🧾 Saves detection results (frame num, track_id, bounding boxes, confidence scores, plate text) to a **CSV file**
- 🎥 Supports both **image** and **video** input formats

---

## 🧩 System Workflow

1. **Input**: Image or video file provided by the user  
2. **Vehicle Detection**: YOLOv8 model identifies and labels vehicles  
3. **License Plate Detection**: A second YOLOv8 model focuses on detecting plates within detected vehicles  
4. **Output**:  
   - Annotated frame or video with bounding boxes  
   - CSV file containing detected vehicle IDs, classes, plate numbers, bounding boxes and confidence scores

---

## 📸 Screenshots

<img width="992" height="847" alt="image" src="https://github.com/user-attachments/assets/04d5d8f2-6e34-4f86-8d03-0e134a42a771" />


