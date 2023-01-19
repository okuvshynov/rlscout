from mnkgame import State, MCTS
import numpy as np
import torch
import sys
import threading
import queue
import multiprocessing

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
threads = multiprocessing.cpu_count()
print(f'Running selfplay in {threads} threads.')

# cli settings
rowsize = 50

def playgame(mcts, boards, probs):
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

torch.save(boards, f'./_out/boards_{rollouts}r_{games}g.pt')
torch.save(probs, f'./_out/probs_{rollouts}r_{games}g.pt')