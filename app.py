import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os

print('Loading video')
cap = cv2.VideoCapture("video4.mp4")

print('Loading cascades')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

print('Starting BG subtractor')
subtractor = cv2.createBackgroundSubtractorKNN()
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))



def findRectangle(image):
	FIRST = 0
	RED = (0, 0, 255)
	THICKNESS = 3
	copy = image.copy()
	ret, thresh = cv2.threshold(copy, 127, 255, 1)
	countours, _ = cv2.findContours(thresh, 1, 2)
	largest = None

	maxCnt = -1
	for contour in countours:
		approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
		if len(approx) > maxCnt:
			maxCnt = len(approx)
		if len(approx) == 4:
			if largest is None or cv2.contourArea(contour) > cv2.contourArea(largest):
				largest = contour
				cv2.drawContours(copy, [largest], FIRST, RED, THICKNESS)

	copy = drawInfo(copy, maxCnt)
	return copy


def preProcessImage(image):
	image2Process = image.copy()
	imageProcessed = cv2.cvtColor(image2Process, cv2.COLOR_BGR2GRAY)
	imageProcessed = cv2.resize(imageProcessed, (640, 480))
	imageProcessed = imageProcessed[100:350, 0:640]
	return imageProcessed

def processBackground(image):
	image2Process = image.copy()
	imageProcessed = subtractor.apply(image2Process)
	imageProcessed = cv2.erode(imageProcessed, element)
	imageProcessed = cv2.dilate(imageProcessed, element)
	numWhitePx = len(imageProcessed[imageProcessed == 255])
	return numWhitePx, imageProcessed

def drawDetectFace(img, img2Print):
	faces, _ = hog.detectMultiScale(img, scale=1.01, winStride=(8, 8), padding=(32, 32), useMeanshiftGrouping=True)
	for (x, y, w, h) in faces:
		pad_w, pad_h = int(0.15 * w), int(0.05 * h)
		cv2.rectangle(img2Print, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (255, 0, 0), 2)
		cv2.rectangle(img, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (255, 0, 0), 2)

def drawInfo(image, info):
	image2Process = image.copy()
	font = cv2.FONT_HERSHEY_SIMPLEX
	imageProcessed = image2Process.copy()
	cv2.putText(imageProcessed, str(info), (10, image.shape[0] - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
	return imageProcessed

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
		print('Read frame fail!')
		time.sleep(.5)
	else:
		frame = preProcessImage(frame)
		numWhitePx, frameWBG = processBackground(frame)
		if(numWhitePx > 10000):
			#frameWBGThreshold = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)
			#ret, thresh = cv2.threshold(frameWBGThreshold, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
			#rev= 255 - thresh
			#contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE)
			#_, threshold = cv2.threshold(frame, 240, 255, cv2.THRESH_BINARY)
			#_, contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			#min_rect_len = 80
			#max_rect_len = 120

			#contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
			#biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
			#cv2.drawContours(frame, [biggest_contour], -1, (255), -1)
			#for cnt in contours:
				#approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
				#cv2.drawContours(frame, [aprox], 0, (0), 5)
				#(x, y, w, h) = cv2.boundingRect(contour)
				#rect = frame[y:y+h, x:x+w]
				#tam = len(rect[rect > 200])
				#if (
				#   h > min_rect_len and
				#   w > min_rect_len and
				#   h < max_rect_len and
				#   w < max_rect_len and
				#   tam > 180 and tam < 250
				#):
				#	cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
				#	cv2.putText(frame, "L: " + str(tam), (x, y + h + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
				#	#cv2.putText(frame, "W: " + str(w) + " - H: " + str(h), (x, y + h + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
			frameWBG = findRectangle(frame)

		frame = drawInfo(frame, numWhitePx)
		numpy_horizontal = np.hstack((frame, frameWBG))
		cv2.imshow('Frames', numpy_horizontal)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		if cv2.waitKey(1) & 0xFF == ord('s'):
			cv2.imwrite('back-open.jpg', frame)
			print("Saved!")

cap.release()
cv2.destroyAllWindows()
