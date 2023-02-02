import duel
from players import AggregatedModelEval, BatchedGamePlayer, CoreMLGameModel, GamePlayer
from game_client import GameClient
from threading import Thread, Lock
import time
import sys
from mnklib import State
import torch
import numpy as np

model_rollouts = 1000
model_temp = 4.0

raw_rollouts = 500000
raw_temp = 1.5

sample_for_n_moves = 8

class ModelStore:
    def __init__(self):
        self.lock = Lock()
        self.model_eval = None
        self.model_id = 0
        self.game_client = GameClient()
        self.batch_size = 4

    # loads new model if different from current
    def maybe_refresh_model(self):
        with self.lock:
            out = self.game_client.get_best_model()
            
            (model_id, torch_model) = out
            if model_id == self.model_id:
                return 
            #print(model_id, torch_model)
            core_ml_model = CoreMLGameModel(torch_model, batch_size=self.batch_size)
            model_eval = AggregatedModelEval(core_ml_model, batch_size=self.batch_size, board_size=8)

            (self.model_id, self.model_eval) = (model_id, model_eval)
            print(f'new best model: {self.model_id}')

    def get_best_model(self):
        self.maybe_refresh_model()
        return (self.model_id, self.model_eval)


def playgame(player, client: GameClient, model_id, board_size, sample_for_n_moves):
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
        if s.winner() == -1:
            outcome = '.'
        if s.winner() == 0:
            outcome = 'a'
        if s.winner() == 1:
            outcome = 'b'

        ## if we were to reuse the search tree, we'd call something like
        # mcts.apply((x, y)) here
        # to move the root to the next node

    return outcome

model_store = ModelStore() 

def selfplay_batch(nthreads, timeout_s=3600):
    start = time.time()
    games_finished = 0
    games_finished_lock = Lock()

    def play_games():
        model_player = BatchedGamePlayer(temp=4.0, rollouts=model_rollouts, model_evaluator=None)
        pure_player = GamePlayer(None, temp=raw_temp, rollouts=raw_rollouts, board_size=8)
        client = GameClient()
        nonlocal games_finished
        while True:
            (model_id, model_eval) = model_store.get_best_model()
            if model_id == 0:
                player = pure_player
            else:
                player = model_player
                player.model_evaluator = model_eval

            result = playgame(player=player, client=client, model_id=model_id, board_size=8, sample_for_n_moves=sample_for_n_moves)
            with games_finished_lock:
                games_finished += 1
            curr = time.time() - start
            sys.stdout.write(result)
            sys.stdout.flush()
            if curr > timeout_s:
                break

    threads = [Thread(target=play_games, daemon=False) for _ in range(nthreads)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    curr = time.time() - start
    print(f'finished {games_finished} games in {curr:.2f} seconds')

selfplay_batch(nthreads=16, timeout_s=3600)