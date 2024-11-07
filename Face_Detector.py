import cv2
import argparse
import time
from random import randrange
import sys
import numpy as np

def init_cascades():
    try:
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
        if face_cascade.empty() or smile_cascade.empty():
            raise Exception("Error loading cascade files")
        return face_cascade, smile_cascade
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

# Initialize parameters
class DetectionParams:
    def __init__(self):
        self.face_scale_factor = 1.1
        self.face_min_neighbors = 5
        self.smile_scale_factor = 1.5
        self.smile_min_neighbors = 10
        self.min_confidence = 0.5

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=int, default=0, help="Video source (default: 0 for webcam)")
    return parser.parse_args()

# Initialize
args = parse_args()
params = DetectionParams()
face_cascade, smile_cascade = init_cascades()

# Initialize video capture
try:
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise Exception("Cannot open video source")
except Exception as e:
    print(f"Error: {str(e)}")
    sys.exit(1)

# Initialize FPS counter
prev_frame_time = 0
new_frame_time = 0

# Initialize face tracking
prev_faces = []

def draw_instructions(frame):
    instructions = [
        "Q: Quit",
        "W/S: Adjust face detection sensitivity",
        "E/D: Adjust smile detection sensitivity",
        "R/F: Adjust minimum neighbors"
    ]
    y = 30
    for text in instructions:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20

while True:
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame")
        break

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(int(fps))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with confidence scores
    faces = face_cascade.detectMultiScale(gray, 
                                        params.face_scale_factor,
                                        params.face_min_neighbors,
                                        minSize=(30, 30),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
    
    # Simple face tracking
    if len(faces) > 0:
        prev_faces = faces
    elif len(prev_faces) > 0:
        faces = prev_faces

    # Process detected faces
    for (x, y, w, h) in faces:
        # Draw face rectangle
        color = (randrange(256), randrange(256), randrange(256))
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Region of interest for smile detection
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect smiles in face region
        smiles = smile_cascade.detectMultiScale(roi_gray,
                                              params.smile_scale_factor,
                                              params.smile_min_neighbors,
                                              minSize=(25, 25))
        
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), 
                         (randrange(256), randrange(256), randrange(256)), 2)

    # Draw FPS and instructions
    cv2.putText(frame, f'FPS: {fps}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    draw_instructions(frame)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1) & 0xFF

    # Key controls
    if key == ord('q'):
        break
    elif key == ord('w'):
        params.face_scale_factor += 0.1
    elif key == ord('s'):
        params.face_scale_factor = max(1.1, params.face_scale_factor - 0.1)
    elif key == ord('e'):
        params.smile_scale_factor += 0.1
    elif key == ord('d'):
        params.smile_scale_factor = max(1.1, params.smile_scale_factor - 0.1)
    elif key == ord('r'):
        params.face_min_neighbors += 1
    elif key == ord('f'):
        params.face_min_neighbors = max(3, params.face_min_neighbors - 1)

# Cleanup
cap.release()
cv2.destroyAllWindows()
print('Application closed successfully')