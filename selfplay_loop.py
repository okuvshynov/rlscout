from mnklib import State
import numpy as np
import torch
import sys
import threading
import queue
import multiprocessing
import time

from players import CoreMLGameModel, GamePlayer

from game_client import GameClient

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

samples_to_keep = 50000

# cli settings
rowsize = 50

print(f'Running selfplay in {threads} threads.')
start = time.time()

class ModelStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.model_id = 0
        self.game_client = GameClient()

    # loads new model if different from current
    def maybe_refresh_model(self):
        with self.lock:
            out = self.game_client.get_best_model()
            
            (model_id, torch_model) = out
            if model_id == self.model_id:
                return 
            #print(model_id, torch_model)
            (self.model_id, self.model) = (model_id, CoreMLGameModel(torch_model.cpu()))
            print(f'new best model: {self.model_id}')

    def get_best_model(self):
        self.maybe_refresh_model()
        return (self.model_id, self.model)

model_store = ModelStore() 

def playgame(player: GamePlayer, client: GameClient, model_id):
    s = State(board_size)
    move_index = 0

    while not s.finished():
        moves = player.get_moves(s)
        # log board state
        board = torch.from_numpy(s.boards()).float()
        
        # log probs
        prob = torch.from_numpy(moves)
        prob = prob / prob.sum()
        
        client.append_sample(board, prob.view(1, board_size, board_size), model_id)
        
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

def playing_thread():
    client = GameClient()
    model_player = GamePlayer(None, temp=model_temp, rollouts=model_rollouts, board_size=8)
    pure_player = GamePlayer(None, temp=raw_temp, rollouts=raw_rollouts, board_size=8)
    while True:
        model_id, model = model_store.get_best_model()
        if model_id != 0:
            model_player.model = model
        current_player = model_player if model_id > 0 else pure_player
        
        try:
            _ = play_queue.get()
        except queue.Empty:
            break
        winner = playgame(current_player, client, model_id)
        client.cleanup_samples(samples_to_keep)

        curr = time.time() - start
        sys.stdout.write(f'{curr:0.2f} {chr_winner[winner]}\n')
        sys.stdout.flush()
        play_queue.task_done()

for g in range(games):
    play_queue.put(g)

for t in range(threads):
    threading.Thread(target=playing_thread, daemon=True).start()

play_queue.join()

print()
print("Done")