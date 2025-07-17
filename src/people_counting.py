import cv2
import numpy as np
import psycopg2
import time
import os
from datetime import datetime
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

# PostgreSQL Connection Configuration
DB_HOST = "localhost"  # Change if hosted remotely
DB_NAME = "postgres"  # Replace with your database name
DB_USER = "postgres"  # Your PostgreSQL username
DB_PASSWORD = "postgres"  # Your PostgreSQL password
DB_PORT = "5432"  # Default PostgreSQL port

# Connect to PostgreSQL
try:
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )
    cursor = conn.cursor()
    print("Connected to PostgreSQL")
except Exception as e:
    print("PostgreSQL Connection Error:", e)
    exit()

# Path
path_video1 = "cam1_1hour.mp4"
path_video2 = "cam2_1hour.mp4"
path_video3 = "cam3_1hour.mp4"
path_video4 = "cam4_1hour.mp4"
path_video5 = "cam5_1hour.mp4"

path_model = "runs/detect/train/weights/best.pt"

# Set confidence score threshold
CONFIDENCE_THRESHOLD = 0.5

# Set AOIs (Area of Interest)
# AOIs = {
#     "Antrian 1": Polygon([(140, 270), (300, 270), (600, 300), (800, 325), 
#                        (1280, 325), (1280, 600), (1100, 650),
#                        (800, 700), (140, 700)]),
#     "Meja 1": Polygon([(140,0), (525,0), (525, 200), (140, 200)]),
#     "Meja 2": Polygon([(770, 0), (1120, 0), (1120, 250), (770, 250)]),
# }

AOIs = {
    "Antrian 2": Polygon([(0, 325), (1280, 325), (1280, 600), (1100, 650),
                       (800, 700), (0, 700)]),
    "Meja 3": Polygon([(770, 0), (1070, 0), (1070, 250), (770, 250)]),
    "Meja 4": Polygon([(330,0), (680,0), (680, 240), (330, 240)])
}

# Load YOLOv11 Model
model = YOLO(path_model)

# Open video file
cap = cv2.VideoCapture(path_video2)

if not cap.isOpened():
    raise IOError("Error opening video file")

# Timer for insert data
last_insert_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection (optimized for GPU)
    results = model.predict(frame, device="cuda")[0]  # Take first result from batch

    # Initialize count dictionary
    counts = {area: 0 for area in AOIs}

    # Extract detections
    boxes = results.boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
    class_ids = results.boxes.cls.cpu().numpy()  # Class IDs
    scores = results.boxes.conf.cpu().numpy()  # Confidence scores

    for box, cls, conf in zip(boxes, class_ids, scores):
        if int(cls) == 0 and conf >= CONFIDENCE_THRESHOLD:  # Class 0 = Person
            x1, y1, x2, y2 = map(int, box)  # Convert to integer
            centroid_x, centroid_y = (x1 + x2) // 2, (y1 + y2) // 2  # Compute centroid
            
            # Default color (green) for bounding box
            bbox_color = (0, 255, 0)  

            # Check if the person is in Area 1
            if AOIs["Antrian 2"].contains(Point(centroid_x, centroid_y)):
                bbox_color = (255, 0, 0)  # Change to blue for Area 1
                counts["Antrian 2"] += 1

            # Check if the person is in Area 2
            if AOIs["Meja 3"].contains(Point(centroid_x, centroid_y)):
                bbox_color = (255, 255, 0)
                counts["Meja 3"] += 1

            # Check if the person is in Area 3
            if AOIs["Meja 4"].contains(Point(centroid_x, centroid_y)):
                bbox_color = (255, 0, 255)
                counts["Meja 4"] += 1

            # Draw bounding box & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)

    # Draw AOIs and display count
    for area_name, polygon in AOIs.items():
        cv2.polylines(frame, [np.array(polygon.exterior.coords, np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(frame, f"{area_name}: {counts[area_name]}", (int(polygon.centroid.x), int(polygon.centroid.y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("YOLOv11 People Counting", frame)

    # Insert data
    current_time = time.time()
    if current_time - last_insert_time >= 5:
        last_insert_time = current_time  # Reset timer
        
        for area_name, count in counts.items():
            if count > 0:  # Insert only if people are detected
                try:
                    cursor.execute(
                        "INSERT INTO people_count (area_name, person_count) VALUES (%s, %s)", 
                        (area_name, count)
                    )
                    conn.commit()
                    print(f"Inserted {count} people in {area_name} into database")
                except Exception as e:
                    conn.rollback()  # Reset transaction to allow further queries
                    print("Error inserting into database:", e)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

cursor.close()
conn.close()
print("PostgreSQL Connection Closed")
print("Current Working Directory:", os.getcwd())  # Add this line in Streamlit

