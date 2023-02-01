import duel
from players import GamePlayer, AggregatedModelEval, BatchedGamePlayer, CoreMLGameModel
from game_client import GameClient
from threading import Thread

client = GameClient() # default localhost:8888

(_, best_model) = client.get_best_model()

core_ml_model = CoreMLGameModel(best_model, batch_size=4)

model_eval = AggregatedModelEval(core_ml_model, batch_size=4, board_size=8)

def play_game():
    A = BatchedGamePlayer(temp=4.0, rollouts=20, model_evaluator=model_eval)
    B = BatchedGamePlayer(temp=4.0, rollouts=20, model_evaluator=model_eval)
    duel.run(A, B, print_board=False)

# start 100 games in 100 threads
nthreads = 8

threads = [Thread(target=play_game) for _ in range(nthreads)]


for t in threads:
    t.start()

for t in threads:
    t.join()

print(model_eval.batch_size_dist)