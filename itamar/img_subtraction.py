import cv2
import numpy as np

font = cv2.FONT_HERSHEY_COMPLEX

img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

scale_percent = 30 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

_, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# for cnt in contours:
#     print(cnt)
#     approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
#     cv2.drawContours(img, [approx], 0, (0), 5)
#     x = approx.ravel()[0]
#     y = approx.ravel()[1]

#     if len(approx) == 4:
#         cv2.putText(img, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0))


cv2.imshow("shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()