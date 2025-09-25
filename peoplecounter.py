import cv2
import threading
import time
from ultralytics import YOLO
from collections import deque
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
import uvicorn
import numpy as np

# Глобальные переменные для API
counter_in = 0
counter_out = 0
current_inside = 0

class Track:
    def __init__(self, id, centroid):
        self.id = id
        self.centroids = deque([centroid], maxlen=30)
        self.crossed_in = False
        self.crossed_out = False

LINE_Y = 200  # y-координата линии на кадре

tracks = {}
track_id_seq = 1

model = YOLO("yolo11n.pt")

def gen_frames(source):
    global counter_in, counter_out, current_inside, tracks, track_id_seq

    cap = cv2.VideoCapture(source)
    fps = 0
    prev = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        person_boxes = []
        for box in results[0].boxes:
            if int(box.cls) == 0:  # класс "person"
                x1, y1, x2, y2 = box.xyxy[0]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                person_boxes.append((cx, cy, int(x1), int(y1), int(x2), int(y2)))

        # Simple tracking (nearest centroid)
        updated_tracks = {}
        used = set()
        for pid, track in tracks.items():
            if len(person_boxes) == 0:
                continue
            last_centroid = track.centroids[-1]
            dists = [np.hypot(last_centroid[0]-bx[0], last_centroid[1]-bx[1]) for bx in person_boxes]
            min_idx = int(np.argmin(dists))
            if min_idx in used or dists[min_idx] > 60:
                continue
            new_centroid = person_boxes[min_idx][:2]
            track.centroids.append(new_centroid)
            updated_tracks[pid] = track
            used.add(min_idx)

        # New tracks
        for i, bx in enumerate(person_boxes):
            if i in used:
                continue
            track = Track(track_id_seq, bx[:2])
            updated_tracks[track_id_seq] = track
            track_id_seq += 1

        tracks = updated_tracks

        # Пересечение линии
        for pid, track in tracks.items():
            if len(track.centroids) < 2:
                continue
            prev_y = track.centroids[-2][1]
            curr_y = track.centroids[-1][1]
            # Вход
            if prev_y < LINE_Y and curr_y >= LINE_Y and not track.crossed_in:
                counter_out += 1
                current_inside = max(current_inside - 1, 0)
                track.crossed_in = True
            # Выход
            if prev_y > LINE_Y and curr_y <= LINE_Y and not track.crossed_out:
                counter_in += 1
                current_inside = max(current_inside + 1, 0)
                track.crossed_out = True

        # Рисуем линию, боксы, счетчики
        cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0,0,255), 2)
        cv2.putText(frame,"INSIDE", (0, LINE_Y-5), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2)
        cv2.putText(frame,"EXIT", (0, LINE_Y+35), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2)
        for pid, track in tracks.items():
            cx, cy = track.centroids[-1]
            cv2.circle(frame, (cx, cy), 6, (0,255,0), -1)
            cv2.putText(frame, f"ID {pid}", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(frame, f"IN: {counter_in}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"OUT: {counter_out}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(frame, f"INSIDE: {current_inside}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        # FPS
        fps = 1/(time.time()-prev)
        prev = time.time()
        cv2.putText(frame, f"FPS: {int(fps)}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

app = FastAPI()

@app.get("/api/counters")
def get_counters():
    return {"in": counter_in, "out": counter_out, "inside": current_inside}

@app.get("/video")
def video_feed(source: str = "0"):
    # source: "0" (вебка), "demo.mp4", "rtmp://..."
    src = int(source) if source.isdigit() else source
    return StreamingResponse(gen_frames(src), media_type="multipart/x-mixed-replace;boundary=frame")

@app.get("/")
def index():
    html = f"""
...
Hello!
...
"""
    return HTMLResponse(content=html)

def run_api():
    uvicorn.run("peoplecounter:app", host="0.0.0.0", port=8069, reload=False)

if __name__ == "__main__":
    run_api()