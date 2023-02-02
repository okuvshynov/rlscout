import duel
from players import AggregatedModelEval, BatchedGamePlayer, CoreMLGameModel
from game_client import GameClient
from threading import Thread, Lock
import time
import sys

client = GameClient() # default localhost:8888

(_, best_model) = client.get_best_model()

def selfplay_batch(nthreads, rollouts, batch_size, timeout_s=300):
    core_ml_model = CoreMLGameModel(best_model, batch_size=batch_size)

    model_eval = AggregatedModelEval(core_ml_model, batch_size=batch_size, board_size=8)

    start = time.time()
    games_finished = 0
    games_finished_lock = Lock()

    def play_games():
        A = BatchedGamePlayer(temp=4.0, rollouts=rollouts, model_evaluator=model_eval)
        B = BatchedGamePlayer(temp=4.0, rollouts=rollouts, model_evaluator=model_eval)
        nonlocal games_finished
        while True:
            result = duel.run(A, B, print_board=False)
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

    print(model_eval.batch_size_dist)
    curr = time.time() - start
    print(f'finished {games_finished} games in {curr:.2f} seconds')