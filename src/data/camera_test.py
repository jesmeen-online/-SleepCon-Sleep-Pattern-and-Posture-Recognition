import cv2
import numpy as np

# Replace this URL with the URL provided by the IP Webcam app on your mobile device
url = 'http://192.168.0.102:8080/video'

# Create a VideoCapture object
cap = cv2.VideoCapture(url)

# Check if the video stream is opened successfully
if not cap.isOpened():
    print("Error: Unable to open video stream")
    exit()

while True:
    ret, frame = cap.read()

    # Check if the frame is received correctly
    if not ret:
        print("Error: Unable to read frame")
        break

    # Display the frame
    cv2.imshow('Mobile Camera Stream', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
