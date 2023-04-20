from src.game_client import GameClient
import torch

client = GameClient()

samples = client.get_batch(1000)
for _, _, b, p, _, _ in samples:
    if torch.isnan(b).any() or torch.isnan(p).any():
        print(b, p)