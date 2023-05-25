import coremltools as ct
import numpy as np
import torch

def to_coreml(torch_model, batch_size=1, board_size=8):
    if torch_model is None:
        return None
    torch_model = torch_model.cpu()
    torch_model.eval()
    sample = torch.ones(batch_size, 2, board_size, board_size)

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
        

        # try get the right output map
        sample = {'x': np.ones((batch_size, 2, board_size, board_size))}
        y = self.model.predict(sample)
        
        ## here we have to figure out which output name corresponds to which output. It's 
        ## unclear how to provide good name, so we just do that based on shape
        for label, value in y.items():
            print(label, value.shape)
            if value.shape == (batch_size, 1):
                self.value_key = label
            elif value.shape == (batch_size, board_size * board_size):
                self.probs_key = label
            else:
                raise Exception('unexpected NN model output shape')



    def get_probs(self, boards):
        sample = {'x': boards.reshape(self.batch_size, 2, self.board_size, self.board_size)}
        y = self.model.predict(sample)

        return np.exp(y[self.probs_key]), y[self.value_key]
