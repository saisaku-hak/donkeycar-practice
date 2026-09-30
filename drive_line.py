import gym
import gym_donkeycar
import numpy as np
import cv2

conf = {"exe_path": "remote", "host": "127.0.0.1", "port": 9091}
env = gym.make("donkey-generated-track-v0", conf=conf)
obs = env.reset()

kp = 1.0
kd = 0.0
offset = -0.3875
lower = np.array([15, 120, 100])
upper = np.array([40, 255, 255])

src = np.float32([[0, 100], [159, 100], [120, 65], [45, 65]])
dst = np.float32([[40, 120], [120, 120], [120, 0], [40, 0]])
M = cv2.getPerspectiveTransform(src, dst)

line_cte = 0.0
prev_error = 0.0
center_x = 80
n_windows = 6
window_height = 120 // n_windows
base_margin = 25
miss_count = 0

for t in range(300):
    img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    warped = cv2.warpPerspective(img, M, (160, 120))
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    current_x = center_x + line_cte * 80
    margin = min(base_margin + miss_count * 15, 80)

    if t == 0:
        full_hist = np.sum(mask[60:120, :], axis=0)
        if full_hist.max() > 300:
            current_x = np.argmax(full_hist)

    found_xs = []

    for w in range(n_windows):
        y_low = 120 - (w + 1) * window_height
        y_high = 120 - w * window_height
        x_low = int(max(0, current_x - margin))
        x_high = int(min(160, current_x + margin))

        band = mask[y_low:y_high, x_low:x_high]
        histogram = np.sum(band, axis=0)

        if histogram.max() > 300:
            local_x = np.argmax(histogram)
            current_x = x_low + local_x
            found_xs.append(current_x)

    if len(found_xs) > 0:
        x = found_xs[0]
        line_cte = (x - center_x) / 80
        miss_count = 0
    else:
        miss_count += 1

    error = line_cte - offset
    d_error = error - prev_error
    steer = np.clip(-kp * error - kd * d_error, -1.0, 1.0)
    prev_error = error

    action = np.array([steer, 0.15])
    obs, reward, done, info = env.step(action)
    print(t, round(line_cte, 3), round(steer, 3), round(margin), round(info["cte"], 2), done)
    if done:
        break

env.close()