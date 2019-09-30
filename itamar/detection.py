import numpy as np
import cv2
import time
import os

cap = cv2.VideoCapture("video/furto1-720p.mp4")
time.sleep(2)

print('Loading cascades')
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def drawDetectFace(img, img2Print):
	#faces = face_cascade.detectMultiScale(img, scaleFactor=1.01, minNeighbors=3, minSize=(150, 150))
	faces, _ = hog.detectMultiScale(img, scale=1.05, padding=(5, 5), winStride=(8, 8))
	for (x, y, w, h) in faces:
		if h > 150:
			pad_w, pad_h = int(0.15 * w), int(0.05 * h)
			cv2.rectangle(img2Print, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (255, 0, 0), 2)
			cv2.rectangle(img, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (255, 0, 0), 2)
			#roi_gray = gray[y:y + h, x:x + w]
			#roi_color = img[y:y + h, x:x + w]

background_image = cv2.imread('back.jpg', cv2.IMREAD_COLOR)
gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)

subtractor = cv2.createBackgroundSubtractorKNN()
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))

print('Starting read')
while(True):
	ret, frame = cap.read()
	if frame is None:
		time.sleep(2)
	else:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = cv2.resize(frame, (640, 480))

		frame2 = np.absolute(frame - gray_background)
		frame2[frame2 > 0] = 255
		frame2 = subtractor.apply(frame)
		frame2 = cv2.erode(frame2, element)
		frame2 = cv2.dilate(frame2, element)
		tamWhite = len(frame2[frame2 == 255])
		# print(tamWhite)
		if(tamWhite > 1000):
			drawDetectFace(frame, frame2)

		cv2.imshow('frame original', frame)
		# cv2.imshow('frame', frame2)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		if cv2.waitKey(1) & 0xFF == ord('s'):
			cv2.imwrite('back.jpg', frame)
			print("Saved!")

cap.release()
cv2.destroyAllWindows()