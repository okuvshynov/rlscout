import matplotlib.pyplot as plt

def plot_sample(board, probs):
    m = board.shape[1]
    n = board.shape[2]
    plt.figure(figsize=(3, 3))
    for x in range(m):
        for y in range(n):
            stone = -1
            if board[0, y, x] > 0:
                stone = 0
            if board[1, y, x] > 0:
                stone = 1

            ch = '0' if stone == 0 else 'X'
            if stone >= 0:
                plt.text(x, y, ch, weight="bold", color="red",
                    fontsize='xx-large', va='center', ha='center')
    plt.imshow(probs.view(m, n).cpu().numpy(), cmap='Blues')
    plt.show()