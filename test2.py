import cv2

'''
img = cv2.imread('aeard-example-before@3x.jpg')
#cv2.imshow("Person",img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#resized = cv2.resize(img, (int(img.shape[1]/2)),int(img.shape[0]/2))
#cv2.imshow("legend",resized)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_img,scaleFactor=1.05,minNeighbors=5)
for x,y,w,h in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
cv2.imshow("Gray", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import time
video = cv2.VideoCapture(0)
check,frame = video.read()
time.sleep(3)
cv2.imshow('Capturing',frame)
cv2.waitKey(0)
video.release()
cv2.destroyAllWindows()
'''
import time

cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop)
ret,frame = cap.read()
while(True):
    cv2.imshow('img1', frame)
    if cv2.waitKey(1) & 0xFF == ord('y'):
        cv2.imwrite('images/c1.png', frame)
        cv2.destroyAllWindows()
        break
cap.release()