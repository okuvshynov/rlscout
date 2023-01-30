from mnklib import MCTS
import numpy as np
import time
from utils import to_coreml
import torch
import copy

# this looks like a wrong level of abstraction. The better one would be to have one for 'model' part only.

class CoreMLGameModel:
    def __init__(self, torch_model, board_size=8):
        self.model = to_coreml(torch_model=torch_model)
        self.board_size = board_size

    def get_probs(self, boards):
        sample = {'x': boards.reshape(1, 2, self.board_size, self.board_size)}
        return np.exp(list(self.model.predict(sample).values())[0])

class TorchGameModel:
    def __init__(self, torch_model, board_size=8):
        self.model = copy.deepcopy(torch_model)
        self.model.eval()
        self.board_size = board_size

    def get_probs(self, boards):
        with torch.no_grad():
            sample = torch.from_numpy(boards).view(1, 2, self.board_size, self.board_size).float()
            return torch.exp(self.model(sample)).numpy().reshape(self.board_size * self.board_size)

class GamePlayer:
    def __init__(self, torch_model, temp=4.0, rollouts=10000, board_size=8):
        self.model = None
        if torch_model is not None:
            self.model = CoreMLGameModel(torch_model, board_size=board_size)

        self.mcts = MCTS(board_size)
        self.temp = temp
        self.rollouts = rollouts
        self.board_size = board_size
        self.thinking_time = 0
        self.moves = 0

    def get_moves(self, state):
        def get_probs(boards, probs_out):
            np.copyto(probs_out, self.model.get_probs(boards))

        get_probs_fn = get_probs if self.model is not None else None

        start = time.time()
        res = self.mcts.run(state, temp=self.temp, rollouts=self.rollouts, get_probs_fn=get_probs_fn)
        self.thinking_time += (time.time() - start)
        self.moves += 1
        return res
    
    def thinking_per_move_ms(self):
        return None if self.moves == 0 else 1000.0 * self.thinking_time / self.moves

    def thinking_per_rollout_ms(self):
        return None if self.moves == 0 else 1000.0 * self.thinking_time / (self.moves * self.rollouts)