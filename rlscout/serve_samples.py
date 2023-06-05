from collections import deque
import time
import zmq
import argparse
import logging
from prometheus_client import Counter
from prometheus_client import start_http_server

from utils.game_db import GameDB

logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/samples_db.log', level=logging.INFO)

parser = argparse.ArgumentParser("sample storage")
parser.add_argument('--port', type=int, default=8889)
parser.add_argument('--db', default='./db/othello6x6_samples.db')
parser.add_argument('-c', '--cleanup_after', default=300000, type=int)
args = parser.parse_args()

port = args.port
db_filename = args.db

# prometheus
calls_counter = Counter('requests', 'sample server requests', labelnames=['method'])
start_http_server(9001)

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

    calls_counter.labels(req['method']).inc()

    # read
    if req['method'] == 'get_batch':
        res['data'] = db.get_batch(req['size'], req['from_id'])

    if req['method'] == 'get_lastn':
        res['data'] = db.get_lastn(req['size'])

    if req['method'] == 'stats':
        res['data'] = db.get_stats()

    # write
    if req['method'] == 'append_sample':
        db.append_sample(req['board'], req['probs'], req['game_id'], req['player'], req['skipped'], req['key'])
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
        
    if queries_processed % 10000 == 0:
        logging.info(f'deleting old samples')
        db.cleanup_samples(args.cleanup_after)

