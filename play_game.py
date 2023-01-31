import duel
from players import GamePlayer
from game_client import GameClient

client = GameClient() # default localhost:8888

pure = GamePlayer(torch_model=None, temp=2.0, rollouts=500000, board_size=8)
(_, best_model) = client.get_best_model()
best = GamePlayer(best_model, temp=4.0, rollouts=300, board_size=8)
#best_10x = GamePlayer(best_model, temp=4.0, rollouts=3000, board_size=8)
#best_100x = GamePlayer(best_model, temp=4.0, rollouts=30000, board_size=8)

duel.run(pure, best, print_board=True)
duel.run(best, pure, print_board=True)

print(f'thinking per move: {pure.thinking_per_move_ms():.3f}ms vs {best.thinking_per_move_ms():.3f}ms')
print(f'thinking per rollout: {pure.thinking_per_rollout_ms():.3f}ms vs {best.thinking_per_rollout_ms():.3f}ms')