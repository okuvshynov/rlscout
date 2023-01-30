from mnklib import State
import numpy as np
from players import GamePlayer

def run(A: GamePlayer, B: GamePlayer, print_board=False):
    players = [A, B]
    s = State(8)
    p = 0

    while not s.finished():
        moves = players[p].get_moves(s)
        # we do no exploration here, just picking move with max visit count/prob
        x, y = np.unravel_index(moves.argmax(), moves.shape)
        s.apply((x, y))
        if print_board:
            s.pp(p, (x, y))
        p = 1 - p

    if s.winner() == -1:
        outcome = '.'
    if s.winner() == 0:
        outcome = 'a'
    if s.winner() == 1:
        outcome = 'b'

    return outcome