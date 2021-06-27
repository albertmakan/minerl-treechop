import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf


class ConvNetwork(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, (8, 8), (4, 4))
        self.conv2 = nn.Conv2d(32, 64, (4, 4), (2, 2))
        self.conv3 = nn.Conv2d(64, 64, (4, 4), (2, 2))
        self.lin1 = nn.Linear(64*2*2, 128)
        self.lin2 = nn.Linear(128, 5)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            assert len(x.shape) == 4, "input shape should be [batch, H, W, C] or [H, W, C]"
        x = x.permute(0, 3, 1, 2) / 255  # BHWC -> BCHW; normalize

        out = nnf.relu(self.conv1(x))
        out = nnf.relu(self.conv2(out))
        out = nnf.relu(self.conv3(out))
        out = out.reshape(out.shape[0], -1)
        out = nnf.relu(self.lin1(out))
        out = self.lin2(out)
        out[:, 2:] = torch.sigmoid(out[:, 2:])
        return out

    def predict(self, x):
        out = self.forward(x)[0]
        action_probabilities = out[2:].detach().numpy()
        sampled_actions = np.random.binomial(1, action_probabilities)
        return {
            "camera": torch.clamp(out[:2], -180, 180).detach().numpy(),
            "forward": sampled_actions[0],
            "jump": sampled_actions[1],
            "attack": sampled_actions[2],
            "sneak": 0,
            "sprint": 0,
            "back": 0,
            "left": 0,
            "right": 0,
        }
