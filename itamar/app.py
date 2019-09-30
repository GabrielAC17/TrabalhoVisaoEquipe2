import numpy as np
import cv2
import time
import os

cap = cv2.VideoCapture("video/720p.mp4")


def draw(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)

    # Remove some small noise if any.
    erode = cv2.erode(thresh, None)
    dilate = cv2.dilate(erode, None)

    # Find contours with cv2.RETR_CCOMP
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_CCOMP, 5)

    for i, cnt in enumerate(contours):
        # Check if it is an external contour and its area is more than 100
        if hierarchy[0, i, 3] == -1 and cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)

            if x > 176 and (x + w) < 393:
                if y > 180 and y < 330 and w > 50 and h > 50 and (y + h) < 330:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    print((x, y, w, h))

    cv2.imshow("shapes", img)
    cv2.imshow("shapes2", dilate)


while(True):
    ret, frame = cap.read()
    if frame is None:
        time.sleep(2)
    else:
        frame = cv2.resize(frame, (640, 480))
        draw(frame)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

    if key & 0xFF == ord(' '):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(
            cap.get(cv2.CAP_PROP_POS_FRAMES)) + 60)

cap.release()
cv2.destroyAllWindows()
