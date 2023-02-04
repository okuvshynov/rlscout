## Brief notes/history of building it

This is a set of notes for myself, so that I can later write it up in a more readable way.

### idea
1. Train a really strong model for some game (currently freestyle gomoku 8x8, maybe othello 8x8 in future) <-- currently here
2. Use that model to guide full search and find a (weak? how to make strong?) solution to a game.

### First implementation of mcts 

At first, MCTS without any model was implemented: https://colab.research.google.com/drive/13Sir3YSGAwZLFJCYaIG4zk83hP9nj9Kk?usp=sharing

It's fairly readable and there are some interesting examples re: what search thinks it should do.

### How to train ML model?

In order to train ML model we'd need to collect training data from our player. Training data will roughly come in the form of list of pairs (board, selected_move) based on what search was selecting. To collect this data we need to let our player play against itself and log the board/move probabilities for each turn.

This would be either really slow or really weak if we move forward with our python implementation, so native state/mcts is implemented 

### Native search/state

We'd like to use it to run initial data collection. The following properties are important:
1. it should be still exploring enough and not having human knowledge baked in
2. it should run in reasonable time
3. it should be of somewhat decent quality

If we run the entire e2e procedure on a more powerful hardware, we can avoid this step. However, for experimentation,  building the whole pipeline and improving cold-start time on conventional hardware it's very useful.

Rough ideas:
1. Store board as a pair of int64 numbers, each representing stones of one player. Current implementation supports boards of up to 8x8 size
2. For search, preallocate buffer of nodes in the search tree in the linear array and reuse it for each search iteration
3. For checking if the game ended, generate a code to compute it with several binary operations. Check if the last move is 'winning'
4. Rather than rewriting entire thing in C++ interface it with Python so that we can use it for convenient visualization.

It can still be improved, of course, but it was good enough


### Training first model

After generating first training data, we split it into training and validation sets and train our first model with Pytorch. Some notes:
1. We use residual tower similar to AlphaGoZero, but with only a few layers (for now) and predicting only probabilities for the next move, no value estimates.
2. I trained it on macbook air with M2 chip using "mps" device, which is short for 'metal pixel shaders' and makes it run on GPU.

### Using the model
After we get the model, we need to use it. The place to use it is within search procedure (which is written at C++ at this point). For experimentation flexibility and visualization we apply model from Python, thus, we needed to make callback from C++ back to Python. 

At this stage it was possible to utilize Apple's Neural Engine - part of their M1-M2 SoC which seem to be good at inference. 

In order to do that torch model is compiled to CoreML model and loaded from Python. 
The data exchange between native library (search) and Python: we use shared numpy arrays as a data buffer, where search will fill in input (boards) and model will fill in our predictions.

At this stage we can run a search with model to guide the search. Model will be used to suggest which paths/notes are worth exploring. (see PUCT)

At the end of each rollout, however, we still do random play as we do not have value model trained yet.

### Getting new training data.
After we confirmed that our new player is stronger than old, model-free search, we can use it to generate new, better training data.

After we collect the data, we can train the new model, and do it again and again. Automating this process requires a little more work though

### 3 processes
There are 3 processes which are running in parallel. They can be distributed or can run on the same host. These processes are:
1. Playing games to collect training data
2. Training a model based on the data we collect, saving snapshots periodically
3. Evaluating the model snapshots we produce, and if they are 'better', let self-play know to use the new one.

To keep track of the housekeeping here we implement game server. Rough idea:
1. Store current state in SQLite
2. Communicate with 3 processes above using 0MQ.

State roughly is:
1. training examples
2. model snapshots
3. the results of model evaluation

At this step we could run 3 processes independently

### Some improvements
While working, it's fairly inefficient at first.
1. We need to clean up old models/samples from db 
2. We need to pass only incremental training data update to trainer
3. Do batching in self-play

### Batching
Batching is really important for model inference on any device, but especially on GPU/ANE. 

Batching is implemented by introducing batching variation of Monte-Carlo Tree Seacrh.
Specifically:
1. Assume single-thread evaluation for now
2. Assume we'd like to have batches of size batch_size
3. Invert the thinking a little - treat games not as 'queries to be processed', but something we can start on demand in order to produce the data.
4. Start batch_size games at once. Allocate MCTS instance for each game.
5. In single thread iterate over the games and run MCTS for each game until we reach node expansion step. At this point we save the context for the search procedure, and expand all batch_size nodes at once, thus, evaluating neural net once with input of 'batch_size' tensors rather than batch_size times with a single tensor. Evaluation is done by making a callback to python where we can load whatever model we'd like.
6. After that, restore the search context and continue back to step 5
7. Sometimes within step (5) game might: pass through expand step if we reached the winning node, or even reach the end of the game. In this case we just restart the game, reset the mcts (while keeping the buffer for nodes) and continue as before. We don't care too much about latency of individual game, we care about throughput.
8. One way to think of it - expand would be like await coroutine.
9. Another way - we can explicitly represent this as state machine and transition games from one state to another.
10. All of the above is done in single CPU thread + evaluates the model on neural engine. On more traditional for ML hardware we'll be able to evaluate on GPU.
11. Now we can start multiple threads doing the same thing.
12. This seems to be easier to scale than building/using multiple-producer-single-consumer queue for individual samples, aggregating them, evaluating and notifying completion/making callbacks/using futures/whatever. We can avoid all thread syncronization (lock free or not) at rollout level and only do some at move level - several orders of magnitude less often.
13. The experiments on M2 CPU/ANE SoC show roughly following results with current setup (2 residual layers, 1000 rollouts per move):
    - multiple threads + no batching -- 14-15 seconds/game
    - multiple threads + batching + queue (with lock) for  individual sampels - 3-4 seconds/game
    - multiple threads + batched MCTS (no high-traffic queue) -- 0.6s/game
14. It is good enough to continue, we can further optimize it when we get to GPU 
15. important: Compared to other methods in literature (e.g. see https://ludii.games/citations/ARXIV2021-1.pdf), as we don't care too much about latency, we are not trying to parallelize/batch individual game state evaluation. Instead, we are running many games at a time and our algorithm is identical to typical sequential MCTS.  

### Next to/write about:

2. show how Apple's Instruments help in understanding performance/utilization of GPU/ANE
3. try on CUDA (rent some GPU machine?)
4. implement Value head for the model. Mention how the fact that the game is a draw affects that.
5. Larger model 
6. check that it works well against some strong players
7. See how well it runs on modern CUDA devices (A100/H100). How to utilize those?
8. Run e2e process for a while to get a strong model

...

9. Start implementing full search for the game of choice 
