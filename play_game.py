import duel
from players import GamePlayer, TorchGameModel, CoreMLGameModel
from game_client import GameClient

client = GameClient() # default localhost:8888

#pure = duel.CoreMLPlayer(torch_model=None, temp=2.0, rollouts=500000, board_size=8)
(_, best_model) = client.get_best_model()
best = GamePlayer(best_model, temp=4.0, rollouts=300, board_size=8)
best_torch = GamePlayer(None, temp=4.0, rollouts=300, board_size=8)
best_torch.model = TorchGameModel(best_model, board_size=8)

duel.run(best_torch, best, print_board=True)
duel.run(best, best_torch, print_board=True)

print(best_torch.thinking_time, best.thinking_time)