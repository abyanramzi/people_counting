import cv2
import numpy as np
import psycopg2
import time
from datetime import datetime
from shapely.geometry import Point, Polygon
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

# PostgreSQL Connection Configuration
DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "postgres",
    "port": "5432",
}

# Video Paths
videos = {
    "video1": "cam1_1hour.mp4",
    "video2": "cam2_1hour.mp4",
    "video3": "cam3_1hour.mp4",
    "video4": "cam6_1hour.mp4",
}

# AOIs for each video
AOIs = {
    "video1": {
        "Antrian 1": Polygon([(140, 270), (300, 270), (600, 300), (800, 325),
                           (1120, 325), (1120, 700), (140, 700)]),
        "Meja 1": Polygon([(140, 0), (525, 0), (525, 200), (140, 200)]),
        "Meja 2": Polygon([(770, 0), (1120, 0), (1120, 250), (770, 250)]),
    },
    "video2": {
        "Antrian 2": Polygon([(140, 325), (1080, 325), (1080, 700), (140, 700)]),
        "Meja 3": Polygon([(330,0), (680,0), (680, 240), (330, 240)]),
        "Meja 4": Polygon([(770, 0), (1070, 0), (1070, 250), (770, 250)]),
    },
    "video3": {
        "Antrian 3": Polygon([(40, 225), (770, 225), (770, 700), (40, 700)]),
        "Meja n": Polygon([(800, 250), (1100, 250), (1100, 500), (800, 500)]),
    },
    "video4": {
        "Meja 5": Polygon([(0,90), (330,90), (330, 360), (0, 360)]),
        "Meja 6": Polygon([(580, 170), (930, 170), (930, 470), (580, 470)]),
        "Meja 7": Polygon([(590, 170), (940, 170), (940, 0), (590, 0)]),
    }
}

# YOLO Model
path_model = "runs/detect/train/weights/best.pt"
model = YOLO(path_model)

# Confidence Threshold
CONFIDENCE_THRESHOLD = 0.3


def process_video(video_name, video_path, aois):
    """ Process a single video stream and perform detection """
    # Connect to PostgreSQL
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print(f"[{video_name}] Connected to PostgreSQL")
    except Exception as e:
        print(f"[{video_name}] PostgreSQL Connection Error:", e)
        return

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[{video_name}] Error opening video file")
        return

    # Timer for database insert
    last_insert_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model.predict(frame, device="cuda")[0]

        # Initialize count dictionary
        counts = {area: 0 for area in aois}

        # Extract detections
        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, class_ids, scores):
            if int(cls) == 0 and conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box)
                centroid_x, centroid_y = (x1 + x2) // 2, (y1 + y2) // 2

                # Default bbox color
                bbox_color = (0, 255, 0)

                for area_name, polygon in aois.items():
                    if polygon.contains(Point(centroid_x, centroid_y)):
                        bbox_color = (255, 0, 0)  # Change color for detection
                        counts[area_name] += 1

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)

        # Draw AOIs and display count
        for area_name, polygon in aois.items():
            cv2.polylines(frame, [np.array(polygon.exterior.coords, np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(frame, f"{area_name}: {counts[area_name]}", 
                        (int(polygon.centroid.x), int(polygon.centroid.y)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow(video_name, frame)

        # Insert data into PostgreSQL
        current_time = time.time()
        if current_time - last_insert_time >= 5:
            last_insert_time = current_time
            for area_name, count in counts.items():
                if count > 0:
                    try:
                        cursor.execute(
                            "INSERT INTO people_count (area_name, person_count, video) VALUES (%s, %s, %s)", 
                            (area_name, count, video_name)
                        )
                        conn.commit()
                        print(f"[{video_name}] Inserted {count} people in {area_name} into database")
                    except Exception as e:
                        conn.rollback()
                        print(f"[{video_name}] Error inserting into database:", e)

        # Exit on key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    cursor.close()
    conn.close()
    print(f"[{video_name}] PostgreSQL Connection Closed")


# Run multiple videos in parallel
with ThreadPoolExecutor(max_workers=len(videos)) as executor:
    for video_name, video_path in videos.items():
        executor.submit(process_video, video_name, video_path, AOIs[video_name])
