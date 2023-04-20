from src.game_client import GameClient
from utils import plot_sample

client = GameClient()

samples = client.get_batch(10)

def format_sample(sample):
    _, _, b, p, _, _= sample
    return b.view(2, 6, 6), p

for s in samples:
    b, p = format_sample(s)
    plot_sample(b, p)