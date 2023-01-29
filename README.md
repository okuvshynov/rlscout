This is an in-progress attempt (with very slow progress) to find a solution to some game. 

Most likely candidate is Othello 8x8.

Rough idea is the following:
1. Use self-play Deep RL (think AlphaZero) to get a very strong model for the game <---- we are here now
2. Use that model to provide a good ordering for full search, like [PVS](https://www.chessprogramming.org/Principal_Variation_Search)

Current state is roughly:
1. 3 core components (Self play, evaluation, model training) are implemented for MNK game (Free Gomoku 8x8). They can run in parallel to each other.
2. pytorch is used for all ML work
3. We use Apple's M2 GPU for training the model ('mps' in pytorch)
4. We use Apple's M2 neural engine for self-play by compiling torch model to apple's coreml
5. We'd like to speed up process a bit but still have diverse enough training data and rely on self-play only. Therefore, a [faster C++ version of state/mcts](mnklib/) is implemented, so that at first iteration we can just run million rollouts and get a somewhat decent player. Once the whole thing is done and we run it on large GPU we can probably avoid doing that, but for now it helps with cold start during experimentation.
6. Only action part of the model is done. We do random rollout at the end for now instead of model-based eval.
7. game server maintains the state of the search. It is backed by SQLite; training, self-play and evaluation loops communicate with it using 0MQ.
8. There's a simple script play_game.py which can be used to see two models/search param sets play against each other.

Immediate next steps:
```
[ ] get rid of all hardcoded constants in the code.
[ ] Implement batching for self-play. We already run multiple search procedures in parallel, but call prediction on batch of size 1. This is very inefficient for any underlying HW (CPU, Apple's Neural Engine, GPU). Aggregate across the self-play and evaluate once instead. This will get more important for larger models.
[ ] Implement value model head.
[ ] Experiment on model architecture/training hyperparams.
[ ] Make it work on cuda as well.
[+] train loop starts from scratch now, need to resume from the model
[ ] cleanup old training samples
[ ] incremental training data update
[ ] do not store symmetries in the db. Generate them on the fly in the training. 
[ ] track time per move in player
```

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
Here's an example of how it looks after 7-10 hours on a single MacBook Air

This is a query which shows 'which model was used to generate self-play data'. 
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
34|27864
37|8456
38|40192
42|20912
44|9280
45|10488
46|21016
```


Here's a history of model selection - if new model snapshot was considered 'better' 
and was selected for self-play (the ones with '+')

```
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
