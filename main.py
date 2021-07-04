import gym
import torch
from train import train
from validation import rollout, evaluate
from wrappers import FrameSkipWrapper

if __name__ == '__main__':
    # train("mpProba", "e:/A/td", learning_rate=0.001, epochs=15)

    model = torch.load("models/model_3c2l_def", map_location=torch.device("cpu"))
    with torch.no_grad():
        reward = rollout(FrameSkipWrapper(gym.make("MineRLTreechop-v0")), model, max_steps=1000, video=True)
    print("reward:", reward)

    # evaluate(model, "e:/A/test_data")
