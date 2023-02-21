from game_client import GameClient
import torch

client = GameClient()

samples = client.get_batch(1000000)
for b, p in samples:
    if torch.isnan(b).any() or torch.isnan(p).any():
        print(b, p)