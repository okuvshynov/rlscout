import queue
import threading
import numpy as np
import time

# backend - something which can get an input of 
# batch_size * 2 * board_size * board_size
# minibatch_size -- something which each worked thread produces, say, 128 samples
# batch_size -- something that model can evaluate efficiently, say, 2048

class BatchEvaluator:
    def __init__(self, backend, minibatch_size, batch_size, board_size):
        self.backend = backend
        self.minibatch_size = minibatch_size
        self.sample_size = 2 * board_size * board_size
        self.batch_size = batch_size
        self.board_size = board_size
        self.minibatches_per_batch = self.batch_size // self.minibatch_size
        self.boards_batch = np.zeros(batch_size * self.sample_size)
        self.q = queue.Queue(batch_size * 4)
        self.batch_size_dist = [0 for _ in range(self.minibatches_per_batch + 1)]
        threading.Thread(target=self.loop, daemon=True).start()
        
    def loop(self):
        probs_outs = [None for _ in range(self.minibatches_per_batch)]
        cvs = [None for _ in range(self.minibatches_per_batch)]

        w = self.sample_size * self.minibatch_size 
        pw = self.board_size * self.board_size * self.minibatch_size
        
        started = time.time()

        while True:
            batch_size = 0
            for i in range(self.minibatches_per_batch):
                try:
                    (minibatch_boards, minibatch_probs_out, cv) = self.q.get(timeout=0.001)
                except queue.Empty as error:
                    break
                self.boards_batch[i*w:(i+1)*w] = minibatch_boards
                probs_outs[i] = minibatch_probs_out
                cvs[i] = cv
                batch_size = i + 1
                self.q.task_done()
            
            if batch_size == 0:
                continue

            probs_batch = self.backend.get_probs(self.boards_batch).reshape((pw * self.minibatches_per_batch, ))
            
            self.batch_size_dist[batch_size] += 1

            if sum(self.batch_size_dist) % 1000 == 0:
                samples = sum([i * self.batch_size_dist[i] for i in range(self.minibatches_per_batch)])
                samples_per_second = samples / (time.time() - started)
                print(time.time() - started, self.batch_size_dist, samples_per_second)

            for i in range(batch_size):
                np.copyto(probs_outs[i], probs_batch[i * pw:(i + 1) * pw].reshape(probs_outs[i].shape))
                with cvs[i]:
                    #print('notifying')
                    cvs[i].notify()

    def enqueue_and_wait(self, boards, probs_out, cv):
        self.q.put((boards, probs_out, cv))
        with cv:
            cv.wait()