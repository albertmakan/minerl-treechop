from datetime import datetime
from PIL import Image
from wrappers import rgb2gray
import torch


def rollout(env, model, max_steps=1e6, video=False, seed=None):
    frames = []
    gray = model.conv1.in_channels == 1
    if seed is not None:
        env.seed(seed)
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    while not done and steps < max_steps:
        action = model.predict(torch.tensor(rgb2gray(obs['pov']) if gray else obs['pov']).float())
        obs, reward, done, info = env.step(action)
        if video:
            frames.append(Image.fromarray(obs['pov']))
        total_reward += reward
        steps += 1
    if video:
        frames[0].save(f"videos/{model.name}_rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif",
                       save_all=True, append_images=frames, duration=50, loop=0)
    return total_reward


def accuracy(expected, predicted):
    return torch.sum(abs(expected[:, 2:]-predicted[:, 2:]) <= 0.5).item(), int(torch.numel(expected)/5*3)

