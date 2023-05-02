from collections import deque
import torch
import logging
import random

# expects tensor of shape [?, N, N], returns list of 8 tensors
def symm(t):
    res = [torch.rot90(t, w, [1, 2]) for w in range(4)]
    t = torch.flip(t, [1])
    res += [torch.rot90(t, w, [1, 2]) for w in range(4)]
    return res

class DataReader:
    def __init__(self, client, dataset_split, device, start_id=0, train_set_rate=0.8, epoch_samples_max=2**20, read_batch_size=2**12):
        self.client = client
        self.dataset_split = dataset_split
        self.current_id = start_id
        self.train_set_rate = train_set_rate
        self.boards_train = None
        self.boards_val = None
        self.probs_train = None
        self.probs_val = None
        self.device = device
        self.epoch_samples_max = epoch_samples_max
        self.read_batch_size = read_batch_size


    def read_samples(self):
        new_samples = deque([], maxlen=self.epoch_samples_max)
        
        batch = self._read_batch()   
        while len(batch) > 0:
            new_samples.extend(batch)
            batch = self._read_batch()
            logging.info(f'{len(new_samples)} samples read.')

        if len(new_samples) == 0:
            return

        nans = [0, 0, 0]

        samples = []

        for _, score, b, p, player, _ in new_samples:
            if torch.isnan(b).any() or torch.isnan(p).any():
                nans[0] += 1
                continue
            if torch.isinf(b).any() or torch.isinf(p).any():
                nans[1] += 1
                continue

            # game was not finished and we didn't record the score
            if score is None:
                nans[2] += 1
                continue

            value = 0 # ignore scores for now

            if player == 1:
                value = - value

            # boards are ordered from POV of current player, but score is
            # from player 0 POV.
            samples.extend(list(zip(symm(b), symm(p), [value] * 8)))
        logging.info(f'observed {nans} corrupted samples out of {len(new_samples)}')

        if not samples:
            return

        random.shuffle(samples)

        boards, probs, scores = zip(*samples)
        scores = list(scores)

        idx = int(self.dataset_split * len(boards))

        boards_train = torch.stack(boards[:idx]).float().to(self.device)
        boards_val = torch.stack(boards[idx:]).float().to(self.device)

        probs_train = torch.stack(probs[:idx]).float().to(self.device)
        probs_val = torch.stack(probs[idx:]).float().to(self.device)

        cat = lambda src, dst: src if dst is None else torch.cat((dst, src), 0)
        self.boards_train = cat(boards_train, self.boards_train) 
        self.boards_val = cat(boards_val, self.boards_val) 
        self.probs_train = cat(probs_train, self.probs_train) 
        self.probs_val = cat(probs_val, self.probs_val) 
            

    def _read_batch(self):
        batch = self.client.get_batch(self.read_batch_size, self.current_id)
        if len(batch) == 0:
            return batch 
        max_id = max(s_id for s_id, _, _, _, _, _ in batch)
        self.current_id = max(max_id, self.current_id)
        return batch
