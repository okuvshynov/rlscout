import zmq
from game_db import GameDB
import time
from collections import deque


port = 8888
db_filename = './_out/8x8/m1.db'
#db_filename = ':memory:'

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:8888")
db = GameDB(db_filename)

queries_processed = 0

samples_last_min = deque()
samples_last_10min = deque()

def append_sample_log():
    now = time.time()
    samples_last_min.append(now)
    samples_last_10min.append(now)

    while samples_last_min[0] + 60.0 < now:
        samples_last_min.popleft()
    while samples_last_10min[0] + 600.0 < now:
        samples_last_10min.popleft()


while True:
    req = socket.recv_json()
    res = {}

    # read
    if req['method'] == 'get_batch':
        res['data'] = [(b, p) for (b, p) in db.get_batch(req['size'])]

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
    if req['method'] == 'append_sample':
        db.append_sample(req['board'], req['probs'], req['model_id'])
        res['data'] = True
        append_sample_log()

    if req['method'] == 'save_model_snapshot':
        db.save_snapshot(req['model'])
        res['data'] = True

    if req['method'] == 'record_eval':
        db.record_evaluation(req['model_id'], req['eval_result'])
        res['data'] = True

    if req['method'] == 'cleanup_samples':
        db.cleanup_samples(req['samples_to_keep'])
        res['data'] = True
    
    socket.send_json(res)

    queries_processed += 1

    if queries_processed % 100 == 0:
        print(f'processed {queries_processed} queries')
        print(f'samples processed: {len(samples_last_min)} last min,  {len(samples_last_10min)} last 10 min')