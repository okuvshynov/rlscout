This is a lazily-but-in-progress attempt to find a solution to some game. 

Most likely candidate is Othello 8x8.

Rough idea is the following:
1. Use self-play DeepRL (think AlphaZero) to get a good model for the game <---- we are here now
2. Use that model to guide a full search (e.g. https://en.wikipedia.org/wiki/Principal_variation_search)

Current state is roughly:
1. 3 components (Self play, evaluation, model training) are implemented for MNK game (Free Gomoku 8x8). They can run in parallel to each other.
2. pytorch is used for all ML work
3. We use Apple's M2 GPU for training the model (called 'mps' in pytorch)
4. We use Apple's M2 neural engine for self-play (by compiling torch model to apple's coreml )
5. To speed up process a bit but still have diverse enough training data and rely on self-play only a faster C++ version of state/mcts is implemented, so that at first iteration we can just run million rollouts and get a somewhat decent player
6. Only action part of the model is done
7. game server maintains the state of the search. It is backed by SQLite; training, self-play and evaluation loops communicate with it using 0MQ.

Immediate next steps:
1. Batching for self-play. we already run multiple search procedures in parallel, but call prediction on batch of size 1. This is very inefficient for any underlying HW (CPU, Apple's Neural Engine, GPU). Aggregate across the self-play and evaluate once instead
2. Implement value model head.
3. Experiment on model
4. Make it work on cuda as well.
5. train loop starts from scratch now, need to resume from the model