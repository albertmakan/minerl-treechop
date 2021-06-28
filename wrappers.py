import minerl
import numpy as np


def rgb2gray(image):
    return image @ np.array([[0.2989], [0.5870], [0.1140]])


class DatasetWrapper:
    def __init__(self, data_dir, gray=False):
        self.data_pipeline = minerl.data.make("MineRLTreechop-v0", data_dir)
        self.gray = gray

    def seq_iter(self, seq_len):
        for state, action, _, _, _ in self.data_pipeline.batch_iter(1, seq_len, 1):
            state = rgb2gray(state['pov'][0]) if self.gray else state['pov'][0]
            yield state, action


def action2array(action_dict, seq_len):
    action_array = np.zeros((seq_len, 5))
    action_array[:, :2] = action_dict['camera']
    action_array[:, 2] = action_dict['forward']
    action_array[:, 3] = action_dict['jump']
    action_array[:, 4] = action_dict['attack']
    return action_array
