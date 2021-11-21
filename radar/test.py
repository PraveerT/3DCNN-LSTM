
while True:
  frame = cap.read()
  print (frame.shape)
  cv2.imshow("frame", frame)
  if chr(cv2.waitKey(1)&255) == 'q':
    break