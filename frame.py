import cv2
import matplotlib.pyplot as plt
from numpy import save

cap = cv2.VideoCapture("http://192.168.100.6:8080/video")
ret,frame = cap.read()
plt.imshow(frame[0:100][0:100])
plt.show()
save('frame.npy',frame[0:100][0:100])
