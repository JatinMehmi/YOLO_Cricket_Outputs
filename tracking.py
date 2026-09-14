import cv2
import numpy as np
import os

def capture_screenshots_at_intervals(video_path, output_dir="screenshots", interval_seconds=0.5):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Fallback standard FPS if metadata is missing
        
    # Calculate how many frames correspond to the half-second interval
    interval_frames = max(1, int(fps * interval_seconds))
    
    frame_count = 0
    saved_count = 0
    
    print(f"Processing video... Saving screenshots every {interval_seconds} seconds ({interval_frames} frames).")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Check if it's time to capture this frame (every 0.5 seconds)
        if frame_count % interval_frames == 0:
            # 1. Run detection on this specific frame
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Multi-color masks (Red + Off-white)
            mask_red1 = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255]))
            mask_red2 = cv2.inRange(hsv, np.array([170, 100, 50]), np.array([180, 255, 255]))
            mask_white = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 50, 255]))
            mask = mask_red1 + mask_red2 + mask_white
            
            # Clean noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_center = None
            best_radius = 0
            
            if len(contours) > 0:
                valid_candidates = []
                for c in contours:
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    area = cv2.contourArea(c)
                    if 3 < radius < 25 and area > 10:
                        valid_candidates.append(((int(x), int(y)), int(radius)))
                
                if valid_candidates:
                    # Pick the largest valid blob match as the ball
                    best_center, best_radius = max(valid_candidates, key=lambda cand: cand[1])

            # 2. Mark the detected point on the frame
            timestamp_sec = round(frame_count / fps, 2)
            if best_center is not None:
                cv2.circle(frame, best_center, best_radius, (0, 255, 0), 2)
                cv2.circle(frame, best_center, 3, (0, 0, 255), -1)
                cv2.putText(frame, f"Ball at {best_center}", (best_center[0] - 40, best_center[1] - best_radius - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add timestamp text to the corner of the screenshot
            cv2.putText(frame, f"Time: {timestamp_sec}s", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # 3. Save the screenshot
            filename = os.path.join(output_dir, f"frame_t_{timestamp_sec}s.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
            print(f"Saved screenshot: {filename}")

        frame_count += 1
        # fjoivef veb;rbgeru
    cap.release()
    print(f"\nFinished! Total {saved_count} screenshots saved in '{output_dir}' folder.")

if __name__ == "__main__":
    video_file = "test.mp4"
    capture_screenshots_at_intervals(video_file, output_dir="ball_screenshots", interval_seconds=0.5)