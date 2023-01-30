import duel
from game_client import GameClient

client = GameClient() # default localhost:8888

pure = duel.CoreMLPlayer(torch_model=None, temp=2.0, rollouts=500000, board_size=8)
(_, best_model) = client.get_best_model()
best = duel.CoreMLPlayer(best_model, temp=4.0, rollouts=3000, board_size=8)

duel.run(pure, best, print_board=True)
duel.run(best, pure, print_board=True)

print(pure.thinking_time, best.thinking_time)