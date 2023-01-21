from mnkgame import State, MCTS
from action_model import ActionNN
import torch
import numpy as np
import sys

chr_winner = {
    -1: '.',
    0: '0',
    1: 'X',
}

# settings
board_size = 7
rollouts = 5000
temp = 1.5
games = 250

# cli settings
rowsize = 50

mcts = MCTS(board_size)

model_path = './_out/model_2000r_100g.pt'

action_model = torch.load(model_path, map_location=torch.device('cpu'))

def mcts_player(s):
    return mcts.run(s, temp=temp, rollouts=rollouts)

def model_player(s):
    board = torch.from_numpy(s.boards()).float()
    board = board.view(1, 2, board_size, board_size)
    logprob = action_model(board)
    board = board.view(2, board_size, board_size)
    valid_moves = torch.ones(board_size, board_size) - (board[0] + board[1])
    probs = torch.exp(logprob).view(board_size, board_size) * valid_moves
    return probs.detach().numpy()

model_wins = 0
for g in range(games):
  s = State(board_size)

  p = g % 2
  players = [model_player, mcts_player]
  wins = [0, 0]

  while not s.finished():
    moves = players[p](s)
    x, y = np.unravel_index(moves.argmax(), moves.shape)
    s.apply((x, y))
    p = 1 - p

  if s.winner() == g % 2:
    model_wins += 1
  sys.stdout.write(f'{chr_winner[s.winner()]}')
  sys.stdout.flush()
  if g % rowsize == rowsize - 1:
    print(f' {g+1} played, {model_wins} model wins')