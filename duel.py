from mnklib import State, MCTS
import numpy as np
import sys
import time

import coremltools as ct
from utils import to_coreml
import torch
from io import BytesIO
from local_db import LocalDB

class CoreMLPlayer:
    def __init__(self, db, model_id, temp=4.0, rollouts=10000, board_size=8):
        torch_model_bin = db.get_model(model_id)
        if torch_model_bin is not None:
            self.model = to_coreml(torch.load(BytesIO(torch_model_bin)).cpu())
        else:
            self.model = None
        self.mcts = MCTS(board_size)
        self.temp = temp
        self.rollouts = rollouts
        self.board_size = board_size
        self.model_id = model_id

    def run(self, state):
        def get_probs(boards, probs):
            sample = {'x': boards.reshape(1, 2, self.board_size, self.board_size)}
            out = np.exp(list(self.model.predict(sample).values())[0])
            np.copyto(probs, out)

        get_probs_fn = get_probs if self.model is not None else None

        return self.mcts.run(state, temp=self.temp, rollouts=self.rollouts, get_probs_fn=get_probs_fn)

def run(db, A: CoreMLPlayer, B: CoreMLPlayer):
    players = [A, B]
    s = State(8)

    players_time = [0, 0]
    p = 0

    while not s.finished():
        start = time.time()
        moves = players[p].run(s)
        players_time[p] += time.time() - start
        # we do no exploration here, just picking move with max visit count/prob
        x, y = np.unravel_index(moves.argmax(), moves.shape)
        s.apply((x, y))
        p = 1 - p

    if s.winner() == -1:
        outcome = '.'
    if s.winner() == 0:
        outcome = 'a'
    if s.winner() == 1:
        outcome = 'b'

    db.log_outcome(A.model_id, A.rollouts, A.temp, B.model_id, B.rollouts, B.temp, outcome, players_time[0], players_time[1])
    return outcome