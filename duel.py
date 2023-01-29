from mnklib import State, MCTS
import numpy as np
import time
from utils import to_coreml

class CoreMLPlayer:
    def __init__(self, torch_model, temp=4.0, rollouts=10000, board_size=8):
        self.model = None
        if torch_model is not None:
            self.model = to_coreml(torch_model)

        self.mcts = MCTS(board_size)
        self.temp = temp
        self.rollouts = rollouts
        self.board_size = board_size

    def run(self, state):
        def get_probs(boards, probs):
            sample = {'x': boards.reshape(1, 2, self.board_size, self.board_size)}
            out = np.exp(list(self.model.predict(sample).values())[0])
            np.copyto(probs, out)

        get_probs_fn = get_probs if self.model is not None else None

        return self.mcts.run(state, temp=self.temp, rollouts=self.rollouts, get_probs_fn=get_probs_fn)

def run(A: CoreMLPlayer, B: CoreMLPlayer):
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

    return outcome