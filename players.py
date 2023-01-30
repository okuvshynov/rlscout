from mnklib import MCTS
import numpy as np
import time
from utils import to_coreml
import torch

class TorchPlayer:
    def __init__(self, torch_model, temp=4.0, rollouts=10000, board_size=8):
        self.model = torch_model
        self.model.eval()

        self.mcts = MCTS(board_size)
        self.temp = temp
        self.rollouts = rollouts
        self.board_size = board_size
        self.thinking_time = 0

    def get_moves(self, state):
        def get_probs(boards, probs):
            with torch.no_grad():
                sample = torch.from_numpy(boards).view(1, 2, self.board_size, self.board_size).float()
                out = torch.exp(self.model(sample)).numpy().reshape(self.board_size * self.board_size)
                np.copyto(probs, out)

        get_probs_fn = get_probs if self.model is not None else None

        start = time.time()
        res = self.mcts.run(state, temp=self.temp, rollouts=self.rollouts, get_probs_fn=get_probs_fn)
        self.thinking_time += (time.time() - start)
        return res

class CoreMLPlayer:
    def __init__(self, torch_model, temp=4.0, rollouts=10000, board_size=8):
        self.model = None
        if torch_model is not None:
            self.model = to_coreml(torch_model)

        self.mcts = MCTS(board_size)
        self.temp = temp
        self.rollouts = rollouts
        self.board_size = board_size
        self.thinking_time = 0

    def get_moves(self, state):
        def get_probs(boards, probs_out):
            sample = {'x': boards.reshape(1, 2, self.board_size, self.board_size)}
            out = np.exp(list(self.model.predict(sample).values())[0])
            np.copyto(probs_out, out)

        get_probs_fn = get_probs if self.model is not None else None

        start = time.time()
        res = self.mcts.run(state, temp=self.temp, rollouts=self.rollouts, get_probs_fn=get_probs_fn)
        self.thinking_time += (time.time() - start)
        return res