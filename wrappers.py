import minerl


class DatasetWrapper:
    def __init__(self, data_dir, gray=False):
        self.data_pipeline = minerl.data.make("MineRLTreechop-v0", data_dir)
        self.gray = gray

    def seq_iter(self, seq_len):
        for state, action, _, _, _ in self.data_pipeline.batch_iter(1, seq_len, 1):
            state = state['pov'][0]
            yield state, action
