import gym
import gym_donkeycar
import numpy as np
from PIL import Image

conf = {"exe_path": "remote", "host": "127.0.0.1", "port": 9091}
env = gym.make("donkey-generated-track-v0", conf=conf)
obs = env.reset()

print(type(obs))
print(obs.shape)
print(obs.dtype)
print(obs.min(), obs.max())

Image.fromarray(obs).save("obs0.png")
row = obs[80]
r = obs[:, :, 0].astype(int)
b = obs[:, :, 2].astype(int)
mask = (r > 130) & (r > b + 20)
print(mask.sum())

vis = obs.copy()
vis[mask] = [255, 0, 255]
Image.fromarray(vis).save("mask0.png")
env.close()