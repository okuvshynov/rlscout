import duel
from players import GamePlayer, AggregatedModelEval, BatchedGamePlayer, CoreMLGameModel
from game_client import GameClient
from threading import Thread
import queue
import time

client = GameClient() # default localhost:8888

(_, best_model) = client.get_best_model()

def selfplay_nobatch(nthreads, ngames, rollouts):
    core_ml_model = CoreMLGameModel(best_model, batch_size=1)
    play_queue = queue.Queue()
    start = time.time()
    
    def play_games():
        A = GamePlayer(torch_model=None, temp=4.0, rollouts=rollouts)
        B = GamePlayer(torch_model=None, temp=4.0, rollouts=rollouts)
        A.model = core_ml_model
        B.model = core_ml_model

        while True:
            try:
                _ = play_queue.get()
            except queue.Empty:
                break
            
            result = duel.run(A, B, print_board=False)
            curr = time.time() - start
            print(f'{curr:.2f} -- {result} -- {A.thinking_per_move_ms()}ms -- {B.thinking_per_move_ms()}ms')
            play_queue.task_done() 
    threads = [Thread(target=play_games, daemon=True) for _ in range(nthreads)]

    for g in range(ngames):
        play_queue.put(g)

    for t in threads:
        t.start()

    play_queue.join()


def selfplay_batch(nthreads, ngames, rollouts, batch_size):
    core_ml_model = CoreMLGameModel(best_model, batch_size=batch_size)

    model_eval = AggregatedModelEval(core_ml_model, batch_size=batch_size, board_size=8)
    play_queue = queue.Queue()

    start = time.time()

    def play_games():
        A = BatchedGamePlayer(temp=4.0, rollouts=rollouts, model_evaluator=model_eval)
        B = BatchedGamePlayer(temp=4.0, rollouts=rollouts, model_evaluator=model_eval)
        while True:
            try:
                _ = play_queue.get()
            except queue.Empty:
                break

            result = duel.run(A, B, print_board=False)
            curr = time.time() - start
            print(f'{curr:.2f} -- {result} -- {A.thinking_per_move_ms()}ms -- {B.thinking_per_move_ms()}ms')
            play_queue.task_done()

    threads = [Thread(target=play_games, daemon=True) for _ in range(nthreads)]

    for g in range(ngames):
        play_queue.put(g)

    for t in threads:
        t.start()

    play_queue.join()

    print(model_eval.batch_size_dist)

#selfplay_nobatch(4, 10, 500)

for batch_size in [1, 2, 4, 8, 16]:
    for nthreads in [batch_size * 2, batch_size * 4]:
        ngames = nthreads * 4
        print(f'b={batch_size} t={nthreads} g={ngames}')
        selfplay_batch(nthreads, ngames=ngames, rollouts=250, batch_size=batch_size)