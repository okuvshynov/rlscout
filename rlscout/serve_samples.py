from collections import deque
import time
import zmq
import argparse
import logging

logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/samples_db.log', encoding='utf-8', level=logging.INFO)

from utils.game_db import GameDB

parser = argparse.ArgumentParser("sample storage")
parser.add_argument('-p', '--port')
parser.add_argument('-d', '--db')
args = parser.parse_args()

port = 8889
db_filename = './db/othello6x6_samples.db'

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
        res['data'] = db.get_batch(req['size'], req['from_id'])

    if req['method'] == 'stats':
        res['data'] = db.get_stats()

    # write
    if req['method'] == 'append_sample':
        db.append_sample(req['board'], req['probs'], req['game_id'], req['player'], req['skipped'])
        res['data'] = True
        append_sample_log()

    if req['method'] == 'cleanup_samples':
        db.cleanup_samples(req['samples_to_keep'])
        res['data'] = True

    if req['method'] == 'game_done':
        db.game_done(req['game_id'], req['score'])
        res['data'] = True
    
    socket.send_json(res)

    queries_processed += 1

    if queries_processed % 100 == 0:
        logging.info(f'processed {queries_processed} queries')
        logging.info(f'samples processed: {len(samples_last_min)} last min,  {len(samples_last_10min)} last 10 min')

