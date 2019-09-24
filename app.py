import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os
import imutils

print('Loading video')
cap = cv2.VideoCapture("video4.mp4")

print('Loading cascades')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

print('Starting BG subtractor')
subtractor = cv2.createBackgroundSubtractorKNN()
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))

def removalNotDarkColor(image):
	image2Process = image.copy()
	#lower_black = np.array([15, 15, 15], dtype='uint16')
	#upper_black = np.array([120, 120, 120], dtype='uint16')
	#imageProcessed = cv2.inRange(image2Process, lower_black, upper_black)
	#imageProcessed = cv2.erode(imageProcessed, element)
	#imageProcessed = cv2.dilate(imageProcessed, element)

	lower_black = np.array([0, 0, 0], dtype='uint16')
	upper_black = np.array([255, 40, 140], dtype='uint16')
	imageProcessed = cv2.cvtColor(image2Process, cv2.COLOR_RGB2HSV)
	imageProcessed = cv2.inRange(imageProcessed, lower_black, upper_black)
	#imageProcessed = cv2.erode(imageProcessed, element)
	#imageProcessed = cv2.dilate(imageProcessed, element)
	return imageProcessed

def findRectangle(image):
	cnts = []
	copy = image.copy()
	copy = cv2.Canny(copy, 50, 100)
	ret, thresh = cv2.threshold(copy, 127, 255, 1)
	countours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

	for contour in countours:
		approx = cv2.convexHull(contour)
		approx = cv2.approxPolyDP(approx, 0.015 * cv2.arcLength(contour, True), True)
		if (
			(len(approx) > 4) and (len(approx) < 10) #and
			#(cv2.contourArea(contour) > 20000.0 and cv2.contourArea(contour) < 80000.0)
		   ):
			cnts.append(approx)

	return cnts

def drawCnt(image, cnt):
	FIRST = 0
	RED = (0, 0, 255)
	THICKNESS = 3
	image2Process = image.copy()
	#approx = cv2.convexHull(contour)
	#approx = cv2.approxPolyDP(approx, 0.02 * cv2.arcLength(contour, True), True)
	cv2.drawContours(image2Process, [cnt], FIRST, RED, THICKNESS)
	return image2Process

def calculateHistogram(image):
	hist = cv2.calcHist([image], [0], None, [256], [0,256])
	hist = hist[200:]
	countFrames = 0
	for values in hist:
		if values > 2000:
			countFrames += values
	imageProcessed = drawInfo(image2Process, countFrames)
	return imageProcessed

def preProcessImage(image, gray=True):
	image2Process = image.copy()
	if gray:
		imageProcessed = cv2.cvtColor(image2Process, cv2.COLOR_BGR2GRAY)
	else:
		imageProcessed = image2Process
	imageProcessed = cv2.resize(imageProcessed, (640, 480))
	imageProcessed = imageProcessed[100:400, 0:640]
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
	ret, frameOriginal = cap.read()
	if frameOriginal is None:
		print('Read frame fail!')
		time.sleep(.5)
	else:
		notGray = preProcessImage(frameOriginal, False)
		frame = preProcessImage(frameOriginal)
		numWhitePx, frameWBG = processBackground(frame)
		if(numWhitePx > 600):
			frameWBG = removalNotDarkColor(notGray)
			frameWBG = (255 - frameWBG)

			cnts = findRectangle(frameWBG)
			frameWBG = frame
			for cnt in cnts:
				frameWBG = drawCnt(frameWBG, cnt)
			frameWBG = cv2.cvtColor(frameWBG, cv2.COLOR_GRAY2BGR)
			# frameWBG = calculateHistogram(frame)

			#frameWBG = frame.copy()
			##edged = cv2.Canny(frame, 50, 100)
			#edged = frame.copy()
			#_, thresh = cv2.threshold(edged, 127, 255, 1)
			#dilate = cv2.dilate(thresh, None)
			#erode = cv2.erode(dilate, None)
			#erode = cv2.bitwise_not(erode)

			#cnts, _ = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			#for c in cnts:
			#	(x, y, w, h) = cv2.boundingRect(c)
			#	if w > 50 and w < 150 and h > 50 and h < 150:
			#		#cv2.drawContours(frameWBG, [c], -1, 0, -1)
			#		cv2.rectangle(frameWBG, (x, x + w), (y, y + h), (255, 0, 0), 1)
			#frame = erode
		else:
			frameWBG = cv2.cvtColor(frameWBG, cv2.COLOR_GRAY2BGR)

		frame = drawInfo(frame, numWhitePx)
		frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

		numpy_horizontal = np.vstack((frame, frameWBG))
		cv2.imshow('Frames', numpy_horizontal)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		if cv2.waitKey(1) & 0xFF == ord('s'):
			cv2.imwrite('back-open.jpg', frame)
			print("Saved!")

cap.release()
cv2.destroyAllWindows()
