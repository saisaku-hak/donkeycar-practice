import cv2
import numpy as np

img = cv2.imread("obs0.png")

src = np.float32([
    [0, 100],
    [159, 100],
    [120, 65],
    [45, 65],
])
dst = np.float32([
    [40, 120],
    [120, 120],
    [120, 0],
    [40, 0],
])

M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(img, M, (160, 120))

hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
lower = np.array([15, 120, 100])
upper = np.array([40, 255, 255])
mask = cv2.inRange(hsv, lower, upper)

histogram = np.sum(mask[60:120, :], axis=0)
x = int(np.argmax(histogram))
print("線の位置 x =", x)
print("ピークの高さ =", histogram[x])