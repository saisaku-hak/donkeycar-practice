import gym
import gym_donkeycar
import numpy as np
import cv2

conf = {"exe_path": "remote", "host": "127.0.0.1", "port": 9091}
env = gym.make("donkey-generated-track-v0", conf=conf)
obs = env.reset()

lower = np.array([15, 120, 100])
upper = np.array([40, 255, 255])

src = np.float32([[0, 100], [159, 100], [120, 65], [45, 65]])
dst = np.float32([[40, 120], [120, 120], [120, 0], [40, 0]])
M = cv2.getPerspectiveTransform(src, dst)

action = np.array([0.0, 0.0])
obs, reward, done, info = env.step(action)

img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
warped = cv2.warpPerspective(img, M, (160, 120))
hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower, upper)

histogram = np.sum(mask[60:120, :], axis=0)
x = int(np.argmax(histogram))
line_cte = (x - 80) / 80

print("実際のcte =", info["cte"])
print("line_cte =", line_cte)
print("差 (line_cte - cte的なもの) =", line_cte)

env.close()