import argparse
import zmq
from prometheus_client import Counter
from prometheus_client import start_http_server

import logging

from utils.game_db import GameDB

logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/models_db.log', level=logging.INFO)

parser = argparse.ArgumentParser("model storage")
parser.add_argument('--port', type=int, default=8888)
parser.add_argument('--db', default='/tmp/othello6x6_models.db')
args = parser.parse_args()

port = args.port
db_filename = args.db

# prometheus
calls_counter = Counter('requests', 'model server requests', labelnames=['method'])
start_http_server(9000)

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(f'tcp://*:{port}')
logging.info(f'listening on port {port}')

db = GameDB(db_filename)
logging.info(f'connected to db {db_filename}')

while True:
    req = socket.recv_json()
    res = {}

    calls_counter.labels(req['method']).inc()

    if req['method'] == 'get_best_model':
        out = db.get_best_model()
        res['data'] = (0, None) if out is None else out

    if req['method'] == 'get_model_to_eval':
        out = db.get_last_not_evaluated_model()
        res['data'] = (0, None) if out is None else out

    if req['method'] == 'count_models_to_eval':
        out = db.count_models_to_eval()
        res['data'] = 0 if out is None else out

    if req['method'] == 'get_last_model':
        out = db.get_last_model()
        res['data'] = (0, None) if out is None else out

    if req['method'] == 'get_model':
        res['data'] = db.get_model(req['id'])

    # write
    if req['method'] == 'save_model_snapshot':
        db.save_snapshot(req['model'])
        res['data'] = True

    if req['method'] == 'record_eval':
        db.record_evaluation(req['model_id'], req['eval_result'])
        res['data'] = True

    socket.send_json(res)
