#!/bin/bash

## simple e2e test of three major components of self-play iterative learning
# requires sqlite3 to run (for now)

rm -f ./db/tmp_models.db
rm -f ./db/tmp_samples.db
rm -rf rlscout/rlslib/_build/

echo "Building native rlslib"
cd rlscout/rlslib && make all && cd ../..

__cleanup ()
{
    pkill -P $$
}
trap __cleanup EXIT

echo "Starting model and training data servers in the background"
python rlscout/serve_models.py --db=./db/tmp_models.db &
python rlscout/serve_samples.py --db=./db/tmp_samples.db &

echo "Starting first iteration of self-play"
python rlscout/selfplay_loop.py --batch_size=64 --games=256 -t 4 --rollouts=500 --random_rollouts=50  2> logs/stderr.log
echo "Starting first iteration of model training"
python rlscout/train_loop.py --snapshots=1 --epoch_samples_min=1000 --minibatch_size=16 --minibatch_per_epoch=5000 2> logs/stderr.log
echo "Starting first iteration of model evaluation"
python rlscout/duel_loop.py --batch_size=32 --games=128 --rollouts=500 --raw_rollouts=500 --iterations=1  2> logs/stderr.log
echo "Starting second iteration of self-play"
python rlscout/selfplay_loop.py --batch_size=64 --games=256 -t 4 --rollouts=500 --random_rollouts=50  2> logs/stderr.log
echo "Starting second iteration of model training"
python rlscout/train_loop.py --snapshots=1 --epoch_samples_min=1000 --minibatch_size=16 --minibatch_per_epoch=5000  2> logs/stderr.log
echo "Starting second iteration of model evaluation"
python rlscout/duel_loop.py --batch_size=32 --games=128 --rollouts=500 --raw_rollouts=500 --iterations=1 2> logs/stderr.log

rows=`sqlite3 db/tmp_models.db 'select id, evaluation from models;' | tr -d ' \t\n\r' `
expected="1|+2|+"

if [ "$rows" == "$expected" ]; then
    echo "Success, two model snapshots created"
    exit 0
else
    echo "ERROR: $rows vs $expected"
    exit 1
fi