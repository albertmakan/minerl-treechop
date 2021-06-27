import numpy as np
import torch
from wrappers import DatasetWrapper, action2array
from model import ConvNetwork


def train(experiment: str, data_path: str, save_path: str, load_path: str = None,
          gray=False, seq_len=64, epochs=10, learning_rate=1e-4):
    data = DatasetWrapper(data_path, gray)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training process started. (Device: {device})")

    if load_path is None:
        model = ConvNetwork(in_channels=1 if gray else 3).to(device)
    else:
        model = torch.load(load_path, map_location=torch.device(device))
        model.eval()
    torch.save(model, save_path + experiment)

    camera_loss = torch.nn.MSELoss()
    actions_loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    mean_errors = []
    for epoch in range(epochs):
        errors = []
        for state, action in data.seq_iter(seq_len):
            print('.', end='')
            model.zero_grad()
            state = torch.tensor(state).float().to(device)
            expected = torch.tensor(action2array(action, seq_len)).float().to(device)
            predicted = model(state)
            loss = camera_loss(predicted[:, :2], expected[:, :2]) + actions_loss(predicted[:, 2:], expected[:, 2:])
            errors.append(loss.cpu().detach().numpy().flatten())
            loss.backward()
            optimizer.step()
        torch.save(model, save_path+experiment)
        mean_errors.append(np.mean(errors))
        print(f"Epoch {epoch}. --- Mean Loss: {mean_errors[epoch]}")
