from mnklib import State, MCTS
import numpy as np
import torch
import sys
import threading
import queue
import multiprocessing
import coremltools as ct
import io

from local_db import LocalDB
from utils import save_sample

chr_winner = {
    -1: '.',
    0: '0',
    1: 'X',
}

# settings
board_size = 8
rollouts = 15
temp = 4.0
sample_for_n_moves = 8
games = 1
threads = 4 # multiprocessing.cpu_count()

coreml_model_path = './_out/8x8/coreml_model_i0_1.mlmodel'

ne_model = ct.models.MLModel(coreml_model_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
def get_probs(boards, probs):
  sample = {'x': boards.reshape(1, 2, board_size, board_size)}
  out = np.exp(list(ne_model.predict(sample).values())[0])
  np.copyto(probs, out)

print(f'Running selfplay in {threads} threads.')

# cli settings
rowsize = 50

localdb = LocalDB('./_out/8x8/test.db')

def playgame(mcts, boards, probs):
  s = State(board_size)
  move_index = 0

  while not s.finished():
    moves = mcts.run(s, temp=temp, rollouts=rollouts, get_probs_fn=get_probs)

    # log board state
    board = torch.from_numpy(s.boards()).float()
    boards.append(board)

    # log probs
    prob = torch.from_numpy(moves)
    prob = prob / prob.sum()
    probs.append(prob)

    save_sample(localdb, board, prob, 1)
    
    # in theory, move selection is based on another 'temperature' parameter
    # which controls the level of exploration by changing the distribution
    # we sample from: 
    # param is tau,  p2 = torch.pow(torch.from_numpy(moves), 1.0 / tau)
    # in practice, however, we either sample from the raw counts/probs
    # or just pick the max greedily.
    if move_index >= sample_for_n_moves:
      x, y = np.unravel_index(moves.argmax(), moves.shape)
    else:
      x, y = np.unravel_index(torch.multinomial(prob.view(-1), 1).item(), prob.shape)

    s.apply((x, y))
    move_index += 1

    ## if we were to reuse the search tree, we'd call something like
    # mcts.apply((x, y)) here
    # to move the root to the next node

  return s.winner()

# this is a substitute for atomic counter
play_queue = queue.Queue()

def playing_thread(boards, probs):
  mcts = MCTS(board_size)
  while True:
    try:
      _ = play_queue.get()
    except queue.Empty:
      break
    winner = playgame(mcts, boards, probs)

    sys.stdout.write(f'{chr_winner[winner]}')
    sys.stdout.flush()
    play_queue.task_done()

for g in range(games):
  play_queue.put(g)

boards, probs = zip(*[([], []) for _ in range(threads)])

for t in range(threads):
  threading.Thread(target=playing_thread, daemon=True, args=(boards[t], probs[t])).start()

play_queue.join()

all_boards = []
all_probs = []

print()
print('Done playing. Joining the data.')
for t in range(threads):
  print(len(boards[t]), len(probs[t]))
  all_boards.extend(boards[t])
  all_probs.extend(probs[t])

boards = torch.stack(all_boards)
probs = torch.stack(all_probs)

print(boards.shape, probs.shape)

torch.save(boards, f'./_out/{board_size}x{board_size}/boards_{rollouts}r_{games}g_i1.pt')
torch.save(probs, f'./_out/{board_size}x{board_size}/probs_{rollouts}r_{games}g_i1.pt')