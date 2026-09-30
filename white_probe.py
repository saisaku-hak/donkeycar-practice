import cv2
import numpy as np

img = cv2.imread("obs0.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

print("右の白線(影あり) (y=100, x=125):", hsv[100, 125])
print("左の白線(明るい) (y=100, x=15):", hsv[100, 15])
print("道路 (y=100, x=80):", hsv[100, 80])