#!/bin/bash

models_db=./db/othello6x6_models10.db
samples_db=./db/othello6x6_samples10.db

rm -f $models_db
rm -f $samples_db
rm -rf rlscout/rlslib/_build/

echo "Building native rlslib"
cd rlscout/rlslib && make all && cd ../..

__cleanup ()
{
    pkill -P $$
}
trap __cleanup EXIT

echo "Starting model and training data servers in the background"
python rlscout/serve_models.py --db=$models_db &
python rlscout/serve_samples.py --db=$samples_db &

echo "Starting self-play in the background"
python rlscout/selfplay_loop.py --batch_size=128 --games=1000000 -t 4 --rollouts=1500 --random_rollouts=100  2> logs/stderr.log &
echo "Starting model training in the background"
python rlscout/train_loop.py --epoch_samples_min=300000 --minibatch_size=64 --minibatch_per_epoch=5000 --epoch_samples_max=1000000 2> logs/stderr.log & 
echo "Starting model evaluation in the background"
python rlscout/duel_loop.py --batch_size=64 --games=128 --rollouts=1500 --raw_rollouts=1500 --random_rollouts=100 2> logs/stderr.log &

while true
do
    echo "Waiting for status update"
    sleep 30
    echo "Current model db status:"
    sqlite3 $models_db 'select id, evaluation from models;'
    echo "Current sample db status:"
    sqlite3 $samples_db 'select count(*) from samples;'
done
