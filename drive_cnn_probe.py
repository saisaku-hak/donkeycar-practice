import gym
import gym_donkeycar
import numpy as np
from tensorflow import keras

conf = {"exe_path": "remote", "host": "127.0.0.1", "port": 9091}
env = gym.make("donkey-generated-track-v0", conf=conf)
obs = env.reset()

model = keras.models.load_model("./models/mypilot3.h5", compile=False)

ts = []
ctes = []

for t in range(500):
    img = np.expand_dims(obs, axis=0).astype(np.float32) / 255.0
    steer, throttle = model.predict(img, verbose=0)
    steer = float(steer[0][0])
    throttle = float(throttle[0][0])

    action = np.array([steer, throttle])
    obs, reward, done, info = env.step(action)
    cte = info["cte"]

    ts.append(t)
    ctes.append(cte)
    print(t, round(cte,2), round(steer,2), round(throttle,2), done)
    if done:
        break

env.close()

import matplotlib.pyplot as plt
plt.plot(ts, ctes)
plt.xlabel("step")
plt.ylabel("cte")
plt.title("CNN driving cte")
plt.savefig("cte_cnn.png")
plt.show()