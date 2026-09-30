import cv2
import numpy as np

img = cv2.imread("obs0.png")

# 台形の4点: 左下, 右下, 右上, 左上
src = np.float32([
    [0, 110],
    [159, 110],
    [115, 65],
    [50, 65],
])

vis = img.copy()
cv2.polylines(vis, [src.astype(np.int32)], isClosed=True, color=(255, 0, 255), thickness=2)
cv2.imwrite("trapezoid.png", vis)