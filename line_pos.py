import cv2
import numpy as np

img = cv2.imread("obs0.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower = np.array([15, 120, 100])
upper = np.array([40, 255, 255])
mask = cv2.inRange(hsv, lower, upper)

for y in [70, 80, 90, 100, 110]:
    row = mask[y]
    xs = np.where(row > 0)[0]
    if len(xs) > 0:
        print(y, np.median(xs))
    else:
        print(y, "なし")