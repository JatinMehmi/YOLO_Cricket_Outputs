import os
import shutil
import cv2
import numpy as np
from collections import deque
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI(title="Cricket Ball Tracker API")

# Add a welcome route so GET / doesn't return a 404
@app.get("/")
async def root():
    return {"message": "Cricket Ball Tracker API is running! Go to /docs to test uploads."}

# Load the YOLO model globally so it stays in memory for fast inference
model = YOLO("best.pt")

@app.post("/track-ball/")
async def track_ball(file: UploadFile = File(...)):
    # Define temporary file paths
    input_path = f"temp_{file.filename}"
    output_path = f"output_slow_{file.filename}"
    
    # Save the uploaded video locally on the server
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Open video capture
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not read the uploaded video file.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        # Slow down settings (2.5x slower)
        slow_motion_factor = 2.5 
        new_fps = original_fps / slow_motion_factor if original_fps > 0 else 30.0

        # Create video writer
        out = cv2.VideoWriter(
            output_path, 
            cv2.VideoWriter_fourcc(*"mp4v"), 
            new_fps, 
            (width, height)
        )

        pts = deque(maxlen=100)
        last_valid_center = None
        HIGH_CONF_THRESHOLD = 0.7  # Strict filter for false positives

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run tracker
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.2, verbose=False)
            current_center = None

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xywh.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                best_box = None
                max_conf = 0.0
                
                # Filter for high-confidence ball detections
                for box, conf in zip(boxes, confidences):
                    if conf > max_conf and conf >= HIGH_CONF_THRESHOLD:
                        max_conf = conf
                        best_box = box
                        
                if best_box is not None:
                    cx, cy, w, h = best_box
                    current_center = (int(cx), int(cy))

            # Interpolate and build the trail
            if current_center is not None:
                if last_valid_center is not None:
                    dist = np.hypot(current_center[0] - last_valid_center[0], current_center[1] - last_valid_center[1])
                    steps = int(max(1, dist / 5)) 
                    for s in range(1, steps + 1):
                        inter_x = int(last_valid_center[0] + (current_center[0] - last_valid_center[0]) * (s / steps))
                        inter_y = int(last_valid_center[1] + (current_center[1] - last_valid_center[1]) * (s / steps))
                        pts.append((inter_x, inter_y))
                else:
                    pts.append(current_center)
                    
                last_valid_center = current_center

            # Draw blue dots
            for pt in pts:
                cv2.circle(frame, pt, int(5), (255, 0, 0), -1)

            out.write(frame)

        cap.release()
        out.release()

    except Exception as e:
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(status_code=500, detail=str(e))

    if os.path.exists(input_path):
        os.remove(input_path)
# comment added for reference
    return FileResponse(output_path, media_type="video/mp4", filename="tracked_output.mp4")