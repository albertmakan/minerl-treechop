from datetime import datetime
from PIL import Image
from wrappers import rgb2gray, DatasetWrapper, action2array
import torch


def rollout(env, model, max_steps=1e6, video=False, seed=None):
    model.eval()
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
        img = frames[0]
        img.save(f"videos/{model.name}_rollout_r{int(total_reward)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif",
                 save_all=True, append_images=frames, duration=50, loop=0)
    return total_reward


def accuracy(expected, predicted):
    return torch.sum(abs(expected[:, 2:] - predicted[:, 2:]) <= 0.5).item(), int(torch.numel(expected) / 5 * 3)


def evaluate(model, test_data_path: str, seq_len=64):
    model.eval()
    gray = model.conv1.in_channels == 1
    data = DatasetWrapper(test_data_path, gray)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing process started. (Device: {device})")
    correct, total = 0, 0
    for state, action in data.seq_iter(seq_len):
        print('.', end='')
        expected = torch.tensor(action2array(action, seq_len)).float().to(device)
        predicted = model(torch.tensor(state).float().to(device))
        c, t = accuracy(expected, predicted)
        correct += c
        total += t
    print(f"\nTest accuracy: {correct}/{total} ({correct / total}%)")
