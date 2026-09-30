import cv2
import numpy as np

img = cv2.imread("obs0.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower = np.array([15, 120, 100])
upper = np.array([40, 255, 255])
mask = cv2.inRange(hsv, lower, upper)

print(mask.sum() // 255)

vis = img.copy()
vis[mask > 0] = [255, 0, 255]
cv2.imwrite("mask_hsv.png", vis)