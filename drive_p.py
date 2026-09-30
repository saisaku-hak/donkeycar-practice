import gym
import gym_donkeycar
import numpy as np

conf = {"exe_path": "remote", "host": "127.0.0.1", "port": 9091}
env = gym.make("donkey-generated-track-v0", conf=conf)
obs = env.reset()

kp = 0.5
kd = 1.0
base_throttle = 0.3
kb = 0.0
cte = 0.0
prev_cte = 0.0

ts = []
ctes = []
steers = []
speeds = []

for t in range(500):
    d_cte = cte - prev_cte
    steer = -kp*cte - kd*d_cte
    steer = np.clip(steer, -1.0, 1.0)
    throttle = base_throttle-kb * abs(cte)
    throttle = np.clip(throttle, 0.0, 1.0)
    action = np.array([steer, throttle])
    obs, reward, done, info = env.step(action)
    prev_cte = cte
    cte = info["cte"]
    print(t, info["speed"], cte, steer, info["hit"], done)
    ts.append(t)
    ctes.append(cte)
    steers.append(steer)
    speeds.append(info["speed"])
    if done:
        break

env.close()

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(ts, ctes, label="cte")
ax1.plot(ts, steers, label="steer")
ax1.legend()
ax2.plot(ts, speeds, label="speed")
ax2.set_xlabel("step")
ax2.legend()
plt.savefig("cte_pd1_speed.png")
plt.show()