import numpy as np
import cv2
import time
import os

cap = cv2.VideoCapture("video4.mp4")
time.sleep(2)

print('Loading cascades')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def drawDetectFace(img, img2Print):
	#faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=(100, 100))
	faces, _ = hog.detectMultiScale(img, scale=1.01, winStride=(8, 8), padding=(32, 32), useMeanshiftGrouping=True)
	#faces, _ = hog.detectMultiScale(img, scale=1.05, padding=(32, 32), winStride=(8, 8))
	for (x, y, w, h) in faces:
		pad_w, pad_h = int(0.15 * w), int(0.05 * h)
		cv2.rectangle(img2Print, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (255, 0, 0), 2)
		cv2.rectangle(img, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (255, 0, 0), 2)
		#roi_gray = gray[y:y + h, x:x + w]
		#roi_color = img[y:y + h, x:x + w]

background_image = cv2.imread('back.jpg', cv2.IMREAD_COLOR)
gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)

background_image_2 = cv2.imread('img1.png', cv2.IMREAD_COLOR)
background_image_2 = cv2.resize(background_image_2, (640, 480))
gray_background_2 = cv2.cvtColor(background_image_2, cv2.COLOR_BGR2GRAY)

subtractor = cv2.createBackgroundSubtractorKNN()
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))

params = cv2.SimpleBlobDetector_Params()
params.filterByCircularity = True
params.minCircularity = 0.7
params.maxCircularity = 0.8
params.filterByArea = 1
params.minArea = 300
params.maxArea = 2000
params.filterByColor = 1
params.blobColor = 0
detector = cv2.SimpleBlobDetector_create(params)

print('Starting read')
while(True):
	ret, frame = cap.read()
	if frame is None:
		time.sleep(2)
	else:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = cv2.resize(frame, (640, 480))

		#frame2 = np.absolute(frame - gray_background)
		#frame2[frame2 > 0] = 255
		frame2 = subtractor.apply(frame)

		frame2 = cv2.erode(frame2, element)
		frame2 = cv2.dilate(frame2, element)
		tamWhite = len(frame2[frame2 == 255])
		print(tamWhite)
		if(tamWhite > 2000):
			frame3 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)
			ret,thresh = cv2.threshold(frame3,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
			rev=255-thresh

			contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE)
			min_rect_len = 80
			max_rect_len = 120

			for contour in contours:
				(x, y, w, h) = cv2.boundingRect(contour)
				rect = frame[y:y+h, x:x+w]
				tam = len(rect[rect > 200])
				if h>min_rect_len and w>min_rect_len and h<max_rect_len and w<max_rect_len and tam > 180 and tam < 250:
					cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
					cv2.putText(frame, "L: " + str(tam), (x, y + h + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
					#cv2.putText(frame, "W: " + str(w) + " - H: " + str(h), (x, y + h + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 

		cv2.imshow('frame original', frame)
		cv2.imshow('frame', frame2)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		if cv2.waitKey(1) & 0xFF == ord('s'):
			cv2.imwrite('back-open.jpg', frame)
			print("Saved!")

cap.release()
cv2.destroyAllWindows()
