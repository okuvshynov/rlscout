from mnkgame import State, MCTS
from action_model import ActionNN
import torch
import numpy as np
import sys
import time

import coremltools as ct

chr_winner = {
    -1: '.',
    0: '0',
    1: 'X',
}

# settings
board_size = 7
rollouts = 10000
temp = 3.0
games = 100

# cli settings
rowsize = 10

mcts = MCTS(board_size)

model_path = './_out/model_cp_1.pt'

action_model = torch.load(model_path, map_location=torch.device('cpu'))

ne_model = ct.models.MLModel(f'./_out/coreml_model_cp_1.mlmodel', compute_units=ct.ComputeUnit.CPU_AND_NE)

def get_probs(boards, probs):
  sample = {'x': boards.reshape(1, 2, board_size, board_size)}
  out = np.exp(list(ne_model.predict(sample).values())[0])
  #probs[1] = 2.0
  #probs[2] = 3.0
  np.copyto(probs, out)
  #print(probs)
  #print(boards)

def mcts_pure_player(s):
    return mcts.run(s, temp=1.5, rollouts=500000)

def mcts_model_player(s):
    return mcts.run(s, temp=5.0, rollouts=1000, get_probs_fn=get_probs)

def model_player(s):
    board = torch.from_numpy(s.boards()).float()
    board = board.view(1, 2, board_size, board_size)
    logprob = action_model(board)
    board = board.view(2, board_size, board_size)
    valid_moves = torch.ones(board_size, board_size) - (board[0] + board[1])
    probs = torch.exp(logprob).view(board_size, board_size) * valid_moves
    return probs.detach().numpy()

model_wins = 0
draws = 0

players_time = [0.0, 0.0]

for g in range(games):
  s = State(board_size)

  p = g % 2
  players = [mcts_model_player, mcts_pure_player]
  wins = [0, 0]
 
  while not s.finished():
    start = time.time()
    moves = players[p](s)
    players_time[p] += time.time() - start
    x, y = np.unravel_index(moves.argmax(), moves.shape)
    s.apply((x, y))
    p = 1 - p

  if s.winner() == -1:
    draws += 1

  if s.winner() == g % 2:
    model_wins += 1
  sys.stdout.write(f'{chr_winner[s.winner()]}')
  sys.stdout.flush()
  if g % rowsize == rowsize - 1:
    print(f' {g+1} played, {model_wins} model wins, {draws} draws, times: {players_time}')