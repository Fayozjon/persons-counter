import cv2
import os
import face_recognition
from ultralytics import YOLO

# --- Настройки ---
VIDEO_SOURCE = "demo.mov"  # mp4, mov, rtmp, 0 для вебкамеры
MODEL_PATH = "yolov8n.pt"  # путь к весам YOLOv8
CONFIDENCE_THRESHOLD = 0.3
KNOWN_FACES_DIR = "stuff"

# --- Загрузка известных лиц ---
known_face_encodings = []
known_face_names = []
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(".jpg"):
        name = filename[:-4]
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)

# --- Класс для трека ---
class Track:
    def __init__(self, pid, centroid, bbox):
        self.pid = pid
        self.centroids = [centroid]
        self.bbox = bbox
        self.crossed_in = False
        self.crossed_out = False
        self.name = "Unknown"

tracks = {}
next_pid = 1

counter_in = 0
counter_out = 0
current_inside = 0

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    LINE_Y = int(height * 0.6)  # Линия чуть ниже центра

    # --- Детекция людей ---
    results = model(frame)[0]
    detections = []
    for r in results.boxes:
        if int(r.cls) == 0 and r.conf > CONFIDENCE_THRESHOLD:  # Класс "person"
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections.append((cx, cy, (x1, y1, x2, y2)))

    # --- Примитивный трекинг по центроиду ---
    tracks = {}
    for det in detections:
        cx, cy, bbox = det
        track = Track(next_pid, (cx, cy), bbox)

        # --- Поиск лица внутри бокса ---
        x1, y1, x2, y2 = bbox
        person_crop = frame[y1:y2, x1:x2]
        face_locations = face_recognition.face_locations(person_crop)
        face_encodings = face_recognition.face_encodings(person_crop, face_locations)
        name = "Unknown"
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                break
        track.name = name

        tracks[next_pid] = track
        next_pid += 1

    # --- Пересечение линии ---
    for pid, track in tracks.items():
        if len(track.centroids) < 2:
            continue
        prev_y = track.centroids[-2][1]
        curr_y = track.centroids[-1][1]
        # Вход
        if prev_y < LINE_Y and curr_y >= LINE_Y and not track.crossed_in:
            counter_in += 1
            current_inside = max(current_inside + 1, 0)
            track.crossed_in = True
        # Выход
        if prev_y > LINE_Y and curr_y <= LINE_Y and not track.crossed_out:
            counter_out += 1
            current_inside = max(current_inside - 1, 0)
            track.crossed_out = True

    # --- Визуализация ---
    cv2.line(frame, (0, LINE_Y), (width, LINE_Y), (0, 0, 255), 2)
    for pid, track in tracks.items():
        cx, cy = track.centroids[-1]
        x1, y1, x2, y2 = track.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Person box
        cv2.circle(frame, (cx, cy), 16, (0, 255, 0), -1)  # Большая зеленая точка
        cv2.putText(frame, f"ID {pid}", (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, track.name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.putText(frame, f"IN: {counter_in}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT: {counter_out}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"INSIDE: {current_inside}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("People Counter with Face Recognition", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()