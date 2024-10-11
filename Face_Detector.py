import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
#img = cv2.imread('RDJ.png')
#img = cv2.imread('image2.png')
# To capture video from webcam
webcam = cv2.VideoCapture(0)

while True:
    
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)))
    
    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key==81 or key==113:
        break

print('Code Completed')