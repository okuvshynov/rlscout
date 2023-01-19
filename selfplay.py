from mnkgame import State, MCTS
import numpy as np
import torch
import sys

chr_winner = {
    -1: '.',
    0: '0',
    1: 'X',
}

# settings
board_size = 7
rollouts = 200000
temp = 1.5
sample_for_n_moves = 8
games = 1000

# cli settings
rowsize = 50

mcts = MCTS(board_size)

boards = []
probs = []

for g in range(games):
  s = State(board_size)

  move_index = 0

  while not s.finished():
    moves = mcts.run(s, temp=temp, rollouts=rollouts)

    # log board state
    board = torch.from_numpy(s.boards()).float()
    boards.append(board)

    # log probs
    prob = torch.from_numpy(moves)
    prob = prob / prob.sum()
    probs.append(prob)

    # in theory, move selection is based on another 'temperature' parameter
    # which controls the level of exploration by changing the distribution
    # we sample from: 
    # param is tau,  p2 = torch.pow(torch.from_numpy(moves), 1.0 / tau)
    # in practice, however, we either sample from the raw counts 
    # or just pick the max greedily.
    if move_index >= sample_for_n_moves:
      x, y = np.unravel_index(moves.argmax(), moves.shape)
    else:
      x, y = np.unravel_index(torch.multinomial(prob.view(-1), 1).item(), prob.shape)

    s.apply((x, y))
    move_index += 1

    ## somewhere here 
    # mcts.apply((x, y))

  sys.stdout.write(f'{chr_winner[s.winner()]}')
  sys.stdout.flush()
  if g % rowsize == rowsize - 1:
    print(f' {g+1} played.')


boards = torch.stack(boards)
probs = torch.stack(probs)

print(boards.shape, probs.shape)

torch.save(boards, "./boards200k.pt")
torch.save(probs, "./probs200k.pt")
