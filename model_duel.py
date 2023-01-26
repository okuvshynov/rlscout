from mnklib import State, MCTS
import numpy as np
import sys
import time

import coremltools as ct

# works with coreml models on apple hw

chr_winner = {
    -1: '.',
    0: '0',
    1: 'X',
}

# cli settings
rowsize = 10

class ModelPlayer:
    def __init__(self, model_path, temp=4.0, rollouts=10000, board_size=8):
        self.model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        self.mcts = MCTS(board_size)
        self.temp = temp
        self.rollouts = rollouts
        self.board_size = board_size

    def run(self, state):
        def get_probs(boards, probs):
            sample = {'x': boards.reshape(1, 2, self.board_size, self.board_size)}
            out = np.exp(list(self.model.predict(sample).values())[0])
            np.copyto(probs, out)

        return self.mcts.run(state, temp=self.temp, rollouts=self.rollouts, get_probs_fn=get_probs)


def duel(board_size, n_games, model_a_path, model_b_path, print_game=False):
    players = [ModelPlayer(model_a_path), ModelPlayer(model_b_path)]

    players_time = [0.0, 0.0]
    a_wins, draws = 0, 0

    for g in range(n_games):
        s = State(board_size)

        p = g % 2
        
        while not s.finished():
            start = time.time()
            moves = players[p].run(s)
            players_time[p] += time.time() - start
            # we do no exploration here, just picking move with max visit count/prob
            x, y = np.unravel_index(moves.argmax(), moves.shape)
            s.apply((x, y))
            if print_game:
                s.pp(p, (x, y))
            p = 1 - p

        if s.winner() == -1:
            draws += 1
        if s.winner() == g % 2:
            a_wins += 1

        if not print_game:
            sys.stdout.write(f'{chr_winner[s.winner()]}')
            sys.stdout.flush()
            if g % rowsize == rowsize - 1:
                print(f' {g+1} played,  A wins: {a_wins}, Draws: {draws}, thinking times: {players_time}')

if __name__ == "__main__":
    duel(board_size=8, n_games=20, print_game=False, model_a_path = './_out/8x8/coreml_model_i1_1.mlmodel', model_b_path = './_out/8x8/coreml_model_i0_1.mlmodel')