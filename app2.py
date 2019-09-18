import numpy as np
import cv2
import time
import os


cap = cv2.VideoCapture("videos/furto3-720p.mp4")



# img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)


# # scale_percent = 30 # percent of original size
# # width = int(img.shape[1] * scale_percent / 100)
# # height = int(img.shape[0] * scale_percent / 100)
# # dim = (width, height)
# # # resize image
# # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)












def draw(img):

	# scale_percent = 30 # percent of original size
	# width = int(img.shape[1] * scale_percent / 100)
	# height = int(img.shape[0] * scale_percent / 100)
	# dim = (width, height)
	# # resize image
	# img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# ret,thresh = cv2.threshold(gray,30,255,1)
	ret,thresh = cv2.threshold(gray,30,255,cv2.THRESH_BINARY_INV)


	# Remove some small noise if any.
	erode = cv2.erode(thresh,None)
	dilate = cv2.dilate(erode,None)

	# Find contours with cv2.RETR_CCOMP
	contours,hierarchy = cv2.findContours(dilate,cv2.RETR_CCOMP,3)

	for i,cnt in enumerate(contours):
		# Check if it is an external contour and its area is more than 100
		if hierarchy[0,i,3] == -1 and cv2.contourArea(cnt)>100:
			x,y,w,h = cv2.boundingRect(cnt)

			if w > 50 and w < 170 and h > 50 and h < 170 and y > 145 and y+h < 400 and x < 530:
				cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

			# m = cv2.moments(cnt)
			# cx,cy = m['m10']/m['m00'],m['m01']/m['m00']
			# cv2.circle(img,(int(cx),int(cy)),3,255,-1)
		
	cv2.imshow("shapes", img)
	cv2.imshow("shapes2", dilate)





	# cv2.imshow("Threshold", img_bin)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()





while(True):
	ret, frame = cap.read()
	if frame is None:
		time.sleep(2)
	else:
		# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = cv2.resize(frame, (640, 480))
		draw(frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()