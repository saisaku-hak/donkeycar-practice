import gym
import gym_donkeycar
import numpy as np

conf = {"exe_path": "remote", "host": "127.0.0.1", "port": 9091}
env = gym.make("donkey-generated-track-v0", conf=conf)
obs = env.reset()

for t in range(200):
    action = np.array([0.3, 0.3])
    obs, reward, done, info = env.step(action)
    print(t, info["speed"], info["cte"], done)
    if done :
        break

env.close()