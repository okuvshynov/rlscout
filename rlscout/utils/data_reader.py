import torch
from utils.game_client import GameClient
from utils.utils import split_int64, symm
import logging

class DataReader:
    def __init__(self, client: GameClient, train_set_rate=0.8, samples_to_query=2**20):
        self.client = client
        self.train_set_cutoff = int(train_set_rate * 256)
        self.samples_to_query = samples_to_query

    def read_samples(self):
        logging.info('querying DB')
        batch = self.client.get_lastn_samples(self.samples_to_query)
        logging.info(f'got {len(batch)} samples from DB')
        nans = [0, 0, 0]

        boards_train = []
        boards_dev = []
        prob_train = []
        prob_dev = []
        scores_train = []
        scores_dev = []

        for _, score, b, p, player, _, key in batch:
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

            value = float(max(-1, min(score, 1)))
            
            # boards are ordered from POV of current player, but score is
            # from player 0 POV.
            if player == 1:
                value = - value
            
            # key is 64 bit int (signed)
            # we add 2**63 to it, treat it as unsigned and split into 8 chunks of 8 bit each.
            keys = split_int64(key)



            for board, prob, score, key in zip(symm(b), symm(p), [value] * 8, keys):
                if key < self.train_set_cutoff:
                    boards_train.append(board)
                    prob_train.append(prob)
                    scores_train.append(torch.tensor(score))
                else:
                    boards_dev.append(board)
                    prob_dev.append(prob)
                    scores_dev.append(torch.tensor(score))

        if not boards_train or not boards_dev:
            return None
        
        return tuple(map(torch.stack, (boards_train, prob_train, scores_train, boards_dev, prob_dev, scores_dev)))
