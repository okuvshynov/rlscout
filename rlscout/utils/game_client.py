from base64 import b64encode, b64decode
from io import BytesIO
import torch
import zmq
import logging

def torch_encode(t):
    if t is None:
        return None
    buf = BytesIO()
    torch.save(t, buf)
    return b64encode(buf.getvalue()).decode('utf-8')

def torch_decode(s):
    if s is None:
        return None
    return torch.load(BytesIO(b64decode(s)), map_location="cpu")

class GameClient:
    def __init__(self, server="tcp://localhost:8888"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        logging.info(f'Connecting to remote server at {server}')
        self.socket.connect(server)

    def get_lastn_samples(self, size):
        req = {
            'method': 'get_lastn',
            'size': size,
        }
        self.socket.send_json(req)
        res = self.socket.recv_json()
        return [(i, v, torch_decode(b), torch_decode(p), player, skipped, key) for (i, v, b, p, player, skipped, key) in res['data']]

    def get_batch(self, size, from_id=0):
        req = {
            'method': 'get_batch',
            'size': size,
            'from_id' : from_id,
        }
        self.socket.send_json(req)
        res = self.socket.recv_json()
        return [(i, v, torch_decode(b), torch_decode(p), player, skipped, key) for (i, v, b, p, player, skipped, key) in res['data']]

    def get_best_model(self):
        req = {
            'method': 'get_best_model'
        }
        self.socket.send_json(req)
        res = self.socket.recv_json()
        (id, model) = res['data']
        if id != 0:
            model = torch_decode(model)
        return (id, model)

    def get_model_to_eval(self):
        req = {
            'method': 'get_model_to_eval'
        }
        self.socket.send_json(req)
        res = self.socket.recv_json()
        (id, model) = res['data']
        if id != 0:
            model = torch_decode(model)
        return (id, model)
    
    def count_models_to_eval(self):
        req = {
            'method': 'count_models_to_eval'
        }
        self.socket.send_json(req)
        res = self.socket.recv_json()
        return res['data']

    def get_last_model(self):
        req = {
            'method': 'get_last_model'
        }
        self.socket.send_json(req)
        res = self.socket.recv_json()
        (id, model) = res['data']
        if id != 0:
            model = torch_decode(model)
        return (id, model)        

    def get_model(self, model_id):
        req = {
            'method': 'get_model',
            'id': model_id
        }
        self.socket.send_json(req)
        res = self.socket.recv_json()
        model = res['data']

        return None if model is None else torch_decode(model)

    def append_sample(self, board_tensor, probs_tensor, game_id, player, skipped, key):
        req = {
            'method': 'append_sample',
            'board': torch_encode(board_tensor),
            'probs': torch_encode(probs_tensor),
            'game_id': game_id,
            'player': player,
            'skipped': skipped,
            'key': key,
        }
        self.socket.send_json(req)
        return self.socket.recv_json()
    
    def game_done(self, game_id, score):
        req = {
            'method': 'game_done',
            'score': score,
            'game_id': game_id
        }
        self.socket.send_json(req)
        return self.socket.recv_json()

    def save_model_snapshot(self, torch_model):
        req = {
            'method': 'save_model_snapshot',
            'model': torch_encode(torch_model)
        }
        self.socket.send_json(req)
        return self.socket.recv_json()

    def record_eval(self, model_id, eval_result):
        req = {
            'method': 'record_eval',
            'model_id': model_id,
            'eval_result': eval_result
        }
        self.socket.send_json(req)
        return self.socket.recv_json()

    def cleanup_samples(self, samples_to_keep):
        req = {
            'method': 'cleanup_samples',
            'samples_to_keep': samples_to_keep
        }
        self.socket.send_json(req)
        return self.socket.recv_json()