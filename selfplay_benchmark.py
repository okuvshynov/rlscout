import duel
from players import GamePlayer, AggregatedModelEval, BatchedGamePlayer, CoreMLGameModel
from game_client import GameClient
from threading import Thread, Lock
import queue
import time
import sys

client = GameClient() # default localhost:8888

(_, best_model) = client.get_best_model()

def selfplay_nobatch(nthreads, rollouts, timeout_s=300):
    core_ml_model = CoreMLGameModel(best_model, batch_size=1)
    start = time.time()

    games_finished = 0
    games_finished_lock = Lock()
    
    def play_games():
        A = GamePlayer(torch_model=None, temp=4.0, rollouts=rollouts)
        B = GamePlayer(torch_model=None, temp=4.0, rollouts=rollouts)
        A.model = core_ml_model
        B.model = core_ml_model
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
    
    curr = time.time() - start
    print(f'finished {games_finished} games in {curr:.2f} seconds')

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

#print('Starting non-batch version')
#for nthreads in [1, 2, 4, 8, 16, 32]:
#    selfplay_nobatch(nthreads=nthreads, rollouts=1000, timeout_s=1200)

## looks like one of these is good enough for now
selfplay_batch(nthreads=16, rollouts=1000, batch_size=4, timeout_s=600)
selfplay_batch(256, rollouts=1000, batch_size=64, timeout_s=1200)
selfplay_batch(128, rollouts=1000, batch_size=32, timeout_s=1200)

## first, nobatch version with different # of threads

#selfplay_batch(512, rollouts=200, batch_size=128, timeout_s=1200)
#selfplay_batch(64, rollouts=200, batch_size=16, timeout_s=1200)

exit(0)


for batch_size in [1, 2, 4, 8, 16]:
    for nthreads in [batch_size * 2, batch_size * 4]:
        ngames = nthreads * 4
        print(f'b={batch_size} t={nthreads} g={ngames}')
        selfplay_batch(nthreads, ngames=ngames, rollouts=200, batch_size=batch_size)