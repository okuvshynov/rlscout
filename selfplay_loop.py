from mnklib import State, MCTS
import numpy as np
import torch
import sys
import threading
import queue
import multiprocessing
import coremltools as ct
from io import BytesIO

from local_db import LocalDB
from utils import save_sample, to_coreml

chr_winner = {
    -1: '.',
    0: '0',
    1: 'X',
}

# settings
board_size = 8

model_rollouts = 1000
model_temp = 4.0

raw_rollouts = 500000
raw_temp = 1.5

sample_for_n_moves = 8
games = 10000
threads = 4 # multiprocessing.cpu_count()


# cli settings
rowsize = 50

db = LocalDB('./_out/8x8/test2.db')

print(f'Running selfplay in {threads} threads.')

class ModelStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.model_id = 0

    # loads new model if different from current
    def maybe_refresh_model(self):
        with self.lock:
            out = db.get_best_model()
            if out is None:
                (self.model_id, self.model) = (0, None)
                return
            (model_id, torch_model_b) = out
            if model_id == self.model_id:
                return 
            torch_model = torch.load(BytesIO(torch_model_b))
            #print(model_id, torch_model)
            (self.model_id, self.model) = (model_id, to_coreml(torch_model.cpu()))

    def get_best_model(self):
        self.maybe_refresh_model()
        return (self.model_id, self.model)

model_store = ModelStore() 


def playgame(mcts, model_id, ne_model):
    s = State(board_size)
    move_index = 0

    def get_probs(boards, probs):
        sample = {'x': boards.reshape(1, 2, board_size, board_size)}
        out = np.exp(list(ne_model.predict(sample).values())[0])
        np.copyto(probs, out)

    while not s.finished():
        if ne_model is not None:
            moves = mcts.run(s, temp=model_temp, rollouts=model_rollouts, get_probs_fn=get_probs)
        else:
            moves = mcts.run(s, temp=raw_temp, rollouts=raw_rollouts)
        # log board state
        board = torch.from_numpy(s.boards()).float()
        
        # log probs
        prob = torch.from_numpy(moves)
        prob = prob / prob.sum()
        
        save_sample(db, board, prob, model_id)
        
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

    db.try_commit()
    return s.winner()

# this is a substitute for atomic counter
play_queue = queue.Queue()

def playing_thread():
  mcts = MCTS(board_size)

  while True:
    model_id, model = model_store.get_best_model()
    try:
      _ = play_queue.get()
    except queue.Empty:
      break
    winner = playgame(mcts, model_id, model)

    sys.stdout.write(f'{chr_winner[winner]}')
    sys.stdout.flush()
    play_queue.task_done()

for g in range(games):
  play_queue.put(g)

for t in range(threads):
  threading.Thread(target=playing_thread, daemon=True).start()

play_queue.join()

print()
print("Done")