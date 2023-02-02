import duel
from players import AggregatedModelEval, BatchedGamePlayer, CoreMLGameModel, GamePlayer
from game_client import GameClient
from threading import Thread, Lock
import time
import sys

model_rollouts = 1000
model_temp = 4.0

raw_rollouts = 500000
raw_temp = 1.5

client = GameClient() # default localhost:8888

class ModelStore:
    def __init__(self):
        self.lock = Lock()
        self.model_eval = None
        self.model_id = 0
        self.game_client = GameClient()
        self.batch_size = 32

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

model_store = ModelStore() 

def selfplay_batch(nthreads, timeout_s=3600):
    start = time.time()
    games_finished = 0
    games_finished_lock = Lock()

    def play_games():
        model_player = BatchedGamePlayer(temp=4.0, rollouts=model_rollouts, model_evaluator=None)
        pure_player = GamePlayer(None, temp=raw_temp, rollouts=raw_rollouts, board_size=8)
        nonlocal games_finished
        while True:
            (model_id, model_eval) = model_store.get_best_model()
            if model_id == 0:
                player = pure_player
            else:
                player = model_player
                player.model_evaluator = model_eval

            result = duel.run(player, player, print_board=False)
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

selfplay_batch(nthreads=128, timeout_s=1800)