import cv2
import numpy as np

img = cv2.imread("obs0.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

row = hsv[80]
s = row[:, 1]
x = int(np.argmax(s))
print("彩度が最大の位置 x =", x, "HSV =", row[x])
print("道路 (x=80) HSV =", row[80])