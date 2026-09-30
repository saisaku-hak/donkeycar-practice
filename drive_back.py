import gym
import gym_donkeycar
import numpy as np

conf = {"exe_path": "remote", "host": "127.0.0.1", "port": 9091}
env = gym.make("donkey-generated-track-v0", conf=conf)
obs = env.reset()

for t in range(20):
    action = np.array([0.0, 0.3])
    obs, reward, done, info = env.step(action)

for t in range(30):
    action = np.array([0.0, -0.5])
    obs, reward, done, info = env.step(action)
    print(t, info["speed"], info["pos"])

env.close()