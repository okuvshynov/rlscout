import coremltools as ct
import numpy as np
import torch

def to_coreml(torch_model, batch_size=1, board_size=8):
    if torch_model is None:
        return None
    torch_model = torch_model.cpu()
    torch_model.eval()
    sample = torch.rand(batch_size, 2, board_size, board_size)

    traced_model = torch.jit.trace(torch_model, sample)
    return ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=sample.shape)],
        compute_units=ct.ComputeUnit.CPU_AND_NE
    )

class EvalBackend:
    def __init__(self, device, torch_model, batch_size=1, board_size=8):
        self.model = to_coreml(torch_model=torch_model, batch_size=batch_size, board_size=board_size)
        self.board_size = board_size
        self.batch_size = batch_size

    def get_probs(self, boards):
        sample = {'x': boards.reshape(self.batch_size, 2, self.board_size, self.board_size)}
        values = self.model.predict(sample)

        #TODO: this is likely wrong. 
        keys = sorted(values.keys())

        return np.exp(list(values[keys[0]])), values[keys[1]]
