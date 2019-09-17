import numpy as np
import cv2
import time
import os
import imutils

cap = cv2.VideoCapture("video/furto2-720p.mp4")


print('Starting read')
while(True):
	ret, frame = cap.read()
	frame = imutils.resize(frame, width=450)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	(thresh, img_bin) = cv2.threshold(frame, 128, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	frame = 255-img_bin 

	cv2.imshow('frame original', img_bin)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()