from game_client import GameClient
from utils import plot_sample

client = GameClient()

samples = client.get_batch(100)

def format_sample(sample):
    b, p = sample
    return b.view(2, 8, 8), p

for s in samples:
    b, p = format_sample(s)
    plot_sample(b, p)