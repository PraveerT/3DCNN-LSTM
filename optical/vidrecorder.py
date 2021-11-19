import cv2
import numpy as np

camera = cv2.VideoCapture("http://192.168.100.6:8080/video")

while(True):
    ret, frame = camera.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()