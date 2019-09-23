import cv2
import numpy as np

cap = cv2.VideoCapture("video/furto3-720p.mp4")

_, img = cap.read()

x = 471
y = 320
w = 134
h = 109

roi = img[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])

term_criteria = (cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, 10, 1)


while True:
    # _, frame = cap.read()
    frame = img
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    ret, track_window = cv2.CamShift(mask, (x, y, w, h), term_criteria)

    pts = ret[0] + ret[1]
    # pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    print(ret)
    print(pts)
    cv2.rectangle(frame, (pts[1], pts[0]),
                  (pts[1] + pts[3], pts[0] + pts[2]), (255, 0, 0), 2)
    # cv2.polylines(frame, [pts], True, (255, 0, 0), 2)

    # cv2.imshow("hsv", hsv)
    # cv2.imshow("mask", mask)

    cv2.imshow("frame", frame)

    key = cv2.waitKey(0)

    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
