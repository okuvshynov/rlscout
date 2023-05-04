#!/bin/bash

rm -f ./db/othello6x6_models2.db
rm -f ./db/othello6x6_samples2.db
rm -rf rlscout/rlslib/_build/

echo "Building native rlslib"
cd rlscout/rlslib && make all && cd ../..

__cleanup ()
{
    pkill -P $$
}
trap __cleanup EXIT

echo "Starting model and training data servers in the background"
python rlscout/serve_models.py --db=./db/othello6x6_models2.db &
python rlscout/serve_samples.py --db=./db/othello6x6_samples2.db &

echo "Starting self-play in the background"
python rlscout/selfplay_loop.py --batch_size=64 --games=65536 -t 4 --rollouts=1500 --random_rollouts=50  2> logs/stderr.log &
echo "Starting model training in the background"
python rlscout/train_loop.py --epoch_samples_min=100000 --minibatch_size=64 --minibatch_per_epoch=5000 --epoch_samples_max=500000 2> logs/stderr.log & 
echo "Starting model evaluation in the background"
python rlscout/duel_loop.py --batch_size=64 --games=128 --rollouts=1500 --raw_rollouts=1500  2> logs/stderr.log &

while true
do
    echo "Waiting for status update"
    sleep 30
    echo "Current model db status:"
    sqlite3 db/othello6x6_models.db 'select id, evaluation from models;'
    echo "Current sample db status:"
    sqlite3 db/othello6x6_samples.db 'select count(*) from samples;'
done
