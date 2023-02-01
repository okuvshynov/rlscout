from mnklib import MCTS
import numpy as np
import time
from utils import to_coreml
import torch
import copy

import queue
import threading

class CoreMLGameModel:
    def __init__(self, torch_model, batch_size=1, board_size=8):
        self.model = to_coreml(torch_model=torch_model, batch_size=batch_size)
        self.board_size = board_size
        self.batch_size = batch_size

    def get_probs(self, boards):
        sample = {'x': boards.reshape(self.batch_size, 2, self.board_size, self.board_size)}
        return np.exp(list(self.model.predict(sample).values())[0])

class TorchGameModel:
    def __init__(self, torch_model, board_size=8):
        self.model = copy.deepcopy(torch_model)
        self.model.eval()
        self.board_size = board_size

    def get_probs(self, boards):
        with torch.no_grad():
            sample = torch.from_numpy(boards).view(1, 2, self.board_size, self.board_size).float()
            return torch.exp(self.model(sample)).numpy().reshape(self.board_size * self.board_size)

class GamePlayer:
    def __init__(self, torch_model, temp=4.0, rollouts=10000, board_size=8):
        self.model = None
        if torch_model is not None:
            self.model = CoreMLGameModel(torch_model, board_size=board_size)

        self.mcts = MCTS(board_size)
        self.temp = temp
        self.rollouts = rollouts
        self.board_size = board_size
        self.thinking_time = 0
        self.moves = 0

    def get_moves(self, state):
        def get_probs(boards, probs_out):
            np.copyto(probs_out, self.model.get_probs(boards))

        get_probs_fn = get_probs if self.model is not None else None

        start = time.time()
        res = self.mcts.run(state, temp=self.temp, rollouts=self.rollouts, get_probs_fn=get_probs_fn)
        self.thinking_time += (time.time() - start)
        self.moves += 1
        return res
    
    def thinking_per_move_ms(self):
        return None if self.moves == 0 else 1000.0 * self.thinking_time / self.moves

    def thinking_per_rollout_ms(self):
        return None if self.moves == 0 else 1000.0 * self.thinking_time / (self.moves * self.rollouts)


class AggregatedModelEval:
    def __init__(self, model, batch_size=4, board_size=8):
        self.model = model
        self.batch_size = batch_size
        self.item_size = 2 * board_size * board_size
        self.boards_batch = np.zeros(batch_size * self.item_size)
        self.q = queue.Queue(batch_size * 4)
        threading.Thread(target=self.loop, daemon=True).start()
        self.batch_size_dist = [0 for _ in range(batch_size + 1)]
    
    def loop(self):
        probs_outs = [None for _ in range(self.batch_size)]
        cvs = [None for _ in range(self.batch_size)] 

        w = self.item_size
        
        while True:
            batch_size = 0
            for i in range(self.batch_size):
                try:
                    (boards, probs_out, cv) = self.q.get(timeout=0.01)
                except queue.Empty as error:
                    break
                self.boards_batch[i*w:(i+1)*w] = boards
                probs_outs[i] = probs_out
                cvs[i] = cv
                batch_size = i + 1
                self.q.task_done()
            # do eval

            if batch_size == 0:
                continue

            probs_batch = self.model.get_probs(self.boards_batch)
            #print(batch_size)
            #print(probs_batch[1].shape)
            self.batch_size_dist[batch_size] += 1
            for i in range(batch_size):
                np.copyto(probs_outs[i], probs_batch[i])
                with cvs[i]:
                    #print('notifying')
                    cvs[i].notify_all()

    def enqueue_and_wait(self, boards, probs_out, cv):
        self.q.put((boards, probs_out, cv))
        with cv:
            #print('waiting')
            cv.wait()


# we'll do batching here on python side. It might be not the most efficient thing ever,
# but once our model is complicated enough the cost of evaluating the model
# will dominate any not-most-efficient synchronization primitives
# We run multiple BatchedGamePlayer in different threads. 
class BatchedGamePlayer:
    def __init__(self, temp=4.0, rollouts=10000, board_size=8, model_evaluator=None):
        self.mcts = MCTS(board_size)
        self.temp = temp
        self.rollouts = rollouts
        self.board_size = board_size
        self.thinking_time = 0
        self.moves = 0
        self.cv = threading.Condition()
        self.model_evaluator = model_evaluator

    def get_moves(self, state):
        def get_probs(boards, probs_out):
            self.model_evaluator.enqueue_and_wait(boards, probs_out, self.cv)

        get_probs_fn = get_probs if self.model_evaluator is not None else None

        start = time.time()
        res = self.mcts.run(state, temp=self.temp, rollouts=self.rollouts, get_probs_fn=get_probs_fn)
        self.thinking_time += (time.time() - start)
        self.moves += 1
        return res
    
    def thinking_per_move_ms(self):
        return None if self.moves == 0 else 1000.0 * self.thinking_time / self.moves

    def thinking_per_rollout_ms(self):
        return None if self.moves == 0 else 1000.0 * self.thinking_time / (self.moves * self.rollouts)