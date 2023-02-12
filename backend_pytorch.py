import copy
import torch
import numpy as np

class EvalBackend:
    def __init__(self, device, torch_model, batch_size=1, board_size=8):
        self.model = copy.deepcopy(torch_model).to(device)
        self.model.eval()
        self.board_size = board_size
        self.batch_size = batch_size
        self.device = device

    def get_probs(self, boards):
        with torch.no_grad():
            sample = torch.from_numpy(boards).view(self.batch_size, 2, self.board_size, self.board_size).float().to(self.device)
            return torch.exp(self.model(sample)).to("cpu").numpy().reshape(self.batch_size * self.board_size * self.board_size)