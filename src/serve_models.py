import argparse
import zmq

import logging

logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/models_db.log', encoding='utf-8', level=logging.INFO)

from game_db import GameDB


parser = argparse.ArgumentParser("model storage")
parser.add_argument('-p', '--port')
parser.add_argument('-d', '--db')
args = parser.parse_args()

port = 8888
db_filename = './db/othello6x6_models.db'

if args.port is not None:
    port = args.port
if args.db is not None:
    db_filename = args.db

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(f'tcp://*:{port}')
logging.info(f'listening on port {port}')

db = GameDB(db_filename)
logging.info(f'connected to db {db_filename}')

while True:
    req = socket.recv_json()
    res = {}

    logging.info(f'request: {req["method"]}')

    if req['method'] == 'get_best_model':
        out = db.get_best_model()
        res['data'] = (0, None) if out is None else out

    if req['method'] == 'get_model_to_eval':
        out = db.get_last_not_evaluated_model()
        res['data'] = (0, None) if out is None else out

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