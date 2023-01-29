import zmq
from game_db import GameDB

port = 8888
db_filename = './_out/8x8/test3.db'
#db_filename = ':memory:'

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:8888")
db = GameDB(db_filename)

queries_processed = 0

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

    if req['method'] == 'get_model':
        res['data'] = db.get_model(req['id'])

    # write
    if req['method'] == 'append_sample':
        db.append_sample(req['board'], req['probs'], req['model_id'])
        res['data'] = True

    if req['method'] == 'save_model_snapshot':
        db.save_snapshot(req['model'])
        res['data'] = True

    if req['method'] == 'record_eval':
        db.record_evaluation(req['model_id'], req['eval_result'])
        res['data'] = True
    
    socket.send_json(res)

    queries_processed += 1

    if queries_processed % 100 == 0:
        print(f'processed {queries_processed} queries')