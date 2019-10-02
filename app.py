import numpy as np
import cv2, os, time
from datetime import datetime

def printWP(message):
	print(datetime.now().strftime("%x") + ' - ' + datetime.now().strftime("%X") + ' | ' + message)

# videoPath = "video4.mp4"
# videoPath = "./TrabalhoVisaoEquipe2/itamar/video/furto2-720p.mp4"
videoPath = "./1080p.mp4"

printWP('Loading video')
cap = cv2.VideoCapture(videoPath)

printWP('Loading cascades')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

printWP('Starting BG subtractor')
subtractor = cv2.createBackgroundSubtractorKNN()
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))

def removalNotDarkColor(image):
	image2Process = image.copy()

	lower_black = np.array([0, 0, 0], dtype='uint16')
	upper_black = np.array([255, 40, 140], dtype='uint16')
	imageProcessed = cv2.cvtColor(image2Process, cv2.COLOR_RGB2HSV)
	imageProcessed = cv2.inRange(imageProcessed, lower_black, upper_black)
	#imageProcessed = cv2.erode(imageProcessed, element)
	#imageProcessed = cv2.dilate(imageProcessed, element)
	return imageProcessed

def findRectangle(image, imageNotProcessed):
	cnts = []
	copy = image.copy()

	copy = cv2.bilateralFilter(copy, 11, 17, 17)
	#copy = cv2.Canny(copy, 50, 255)
	ret, thresh = cv2.threshold(copy, 50, 255, cv2.THRESH_BINARY)
	contours, hierarchy = cv2.findContours(copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#countours = sorted(countours, key = cv2.contourArea, reverse = True)[:10]

	hull = []
	for contour in contours:
		area = cv2.contourArea(contour)
		if (
			(area > 1000.0 and area < 8000.0)
		):
			rect = cv2.boundingRect(contour)

			x, y, w, h = rect
			roi = imageNotProcessed[x:x + w, y:y + h]
			roi = cv2.equalizeHist(roi)
			hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
			nPxDarkness = sum(hist[250:])
			nPxTotal = sum(hist)
			if nPxTotal > 0:
				avg = sum(hist[230:]) / float(len(hist[230:]))
				percentPxDarkness = (nPxDarkness / nPxTotal) * 100
				if avg > 50:
					#print(str(nPxDarkness) + ' / ' + str(nPxTotal) + ' = ' + str(percentPxDarkness) + '%')
					hull.append(cv2.convexHull(contour, False))
	return hull, hierarchy

	for contour in contours:
		approx = cv2.convexHull(contour)
		approx = cv2.approxPolyDP(approx, 0.015 * cv2.arcLength(contour, True), True)
		if (
			#(len(approx) >= 4) and (len(approx) < 10) and
			(cv2.contourArea(contour) > 1.0 and cv2.contourArea(contour) < 2.0)
		   ):
			cnts.append(approx)

	return cnts

def drawCnt(image, cnts, hierarchy):
	RED = (0, 0, 255)
	THICKNESS = 3
	image2Process = image.copy()
	for i, cnt in enumerate(cnts):
		cv2.drawContours(image2Process, cnts, i, RED, THICKNESS, 8)
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

startTime = datetime.now()
printWP('Iniciando leitura do arquivo: ' + videoPath)
while(True):
	try:
		ret, frameOriginal = cap.read()
		if frameOriginal is None:
			printWP('Fim do vídeo!')
			break
			time.sleep(.5)
		else:
			notGray = preProcessImage(frameOriginal, False)
			frame = preProcessImage(frameOriginal)
			cv2.rectangle(frame, (84, 0), (153, 84), (255, 255, 255), -1)
			cv2.rectangle(notGray, (84, 0), (153, 84), (255, 255, 255), -1)
			numWhitePx, frameWBG = processBackground(frame)
			if(numWhitePx > 600):
				frameWBG = removalNotDarkColor(notGray)
				frameWBG = (255 - frameWBG)

				cnts, hierarchy = findRectangle(frameWBG, frame)
				if cnts and len(cnts) > 0:
					frameWBG = drawCnt(frame, cnts, hierarchy)
					elapsed = datetime.now() - startTime
					printWP('Detectado PC Roubado: ' + str(elapsed))
				else:
					frameWBG = frame

			#frame = drawInfo(frame, numWhitePx)
			#frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
			frame = cv2.cvtColor(notGray, cv2.COLOR_BGR2HSV)
			frameWBG = cv2.cvtColor(frameWBG, cv2.COLOR_GRAY2BGR)

			numpy_horizontal = np.vstack((frame, frameWBG))
			#cv2.imshow('Frames', numpy_horizontal)
			#if cv2.waitKey(1) & 0xFF == ord('q'):
			#	break
			#if cv2.waitKey(1) & 0xFF == ord('s'):
			#	cv2.imwrite('back-open.jpg', frame)
			#	print("Saved!")
	except Exception as e:
		print('Erro: ' + repr(e))
		break

cap.release()
cv2.destroyAllWindows()

