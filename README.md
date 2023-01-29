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
5. We'd like to speed up process a bit but still have diverse enough training data and rely on self-play only. Therefore, a faster C++ version of state/mcts is implemented, so that at first iteration we can just run million rollouts and get a somewhat decent player
6. Only action part of the model is done
7. game server maintains the state of the search. It is backed by SQLite; training, self-play and evaluation loops communicate with it using 0MQ.

Immediate next steps:
1. get rid of all hardcoded constants in the code.
2. Implement batching for self-play. We already run multiple search procedures in parallel, but call prediction on batch of size 1. This is very inefficient for any underlying HW (CPU, Apple's Neural Engine, GPU). Aggregate across the self-play and evaluate once instead. This will get more important for larger models.
3. Implement value model head.
4. Experiment on model architecture/training hyperparams.
5. Make it work on cuda as well.
6. train loop starts from scratch now, need to resume from the model
7. cleanup old training samples
8. incremental training data update
9. do not store symmetries in the db. Generate them on the fly in the training. 
10. track time per move in player


Current way to run the process:
1. start game server: 

```% python game_server.py   # <-- modify the path to sqlite db file if needed```

2. start self-play: 

```python selfplay_loop.py   # <-- it will start playing 'no model' mcts with 500k rollouts```

3. start model training:

```python train_loop.py # <-- it will wait till it gets enough initial samples```

4. start model eval: 

```python duel_loop.py```

To monitor what's going on we can query sqlite db.
Here's an example of how it looks after 7-8 hours:
```
% sqlite3 ./_out/8x8/test3.db
sqlite> select produced_by_model, sum(1) from samples group by produced_by_model;
0|166384
3|39736
8|22632
10|20080
12|10944
13|44032
17|10504
18|12880
19|22112
21|21856
24|74584
33|9288
34|9496


sqlite> select id, evaluation from models;
1|-
2|-
3|+
4|-
5|-
6|-
7|-
8|+
9|-
10|+
11|-
12|+
13|+
14|-
15|-
16|-
17|+
18|+
19|+
20|-
21|+
22|-
23|-
24|+
25|-
26|-
27|-
28|-
29|-
30|-
31|-
32|-
33|+
34|+
35|-
```
