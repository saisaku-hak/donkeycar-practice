import gym
import gym_donkeycar
import numpy as np

conf = {"exe_path": "remote", "host": "127.0.0.1", "port": 9091}
env = gym.make("donkey-generated-track-v0", conf=conf)
obs = env.reset()

kp = 1.0
kd = 1.0
cte = 0.0
prev_cte = 0.0

stuck_count = 0
backing_up = 0

for t in range(500):
    if backing_up > 0:
        steer = np.clip(-0.5 * np.sign(cte), -1.0, 1.0)
        throttle = -0.5
        backing_up -= 1
    else:
        d_cte = cte - prev_cte
        steer = -kp*cte - kd*d_cte
        steer = np.clip(steer, -1.0, 1.0)
        throttle = 0.15

    action = np.array([steer, throttle])
    obs, reward, done, info = env.step(action)
    prev_cte = cte
    cte = info["cte"]
    speed = info["speed"]

    if throttle > 0 and speed < 0.1:
        stuck_count += 1
    else:
        stuck_count = 0

    if stuck_count > 5 and backing_up == 0:
        backing_up = 20
        stuck_count = 0

    if abs(cte) > 4 and backing_up == 0:
        backing_up = 20
        stuck_count = 0

    print(t, round(speed,2), round(cte,2), round(steer,2), round(throttle,2), backing_up, done)
    if done:
        break

env.close()