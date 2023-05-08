#!/bin/bash

__cleanup ()
{
    pkill -P $$
}
trap __cleanup EXIT

__log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') $1"
}

rm -f ./db/models_a.db
rm -f ./db/samples_a.db
rm -f ./db/models_b.db
rm -f ./db/samples_b.db
rm -rf rlscout/rlslib/_build/

__log "Building native rlslib"
    cd rlscout/rlslib && make all && cd ../..

__log " === Testing A === "

__log "Starting model and training data servers in the background"
python rlscout/serve_models.py --db=./db/models_a.db &
python rlscout/serve_samples.py --db=./db/samples_a.db &

__log "Starting first iteration of self-play"
python rlscout/selfplay_loop.py --batch_size=64 --games=64 -t 1 --rollouts=500 --random_rollouts=20
__log "Starting first iteration of model training"
python rlscout/train_loop.py --snapshots=1 --epoch_samples_min=500 --minibatch_size=16 --minibatch_per_epoch=5000 2> logs/stderr.log
__log "Starting first iteration of model evaluation"
python rlscout/duel_loop.py --batch_size=32 --games=32 --rollouts=500 --raw_rollouts=500 --iterations=1 --random_rollouts=20 2> logs/stderr.log

__log "Starting second iteration of self-play"
python rlscout/selfplay_loop.py --batch_size=64 --games=64 -t 1 --rollouts=500 --random_rollouts=20  2> logs/stderr.log
__log "Starting second iteration of model training"
python rlscout/train_loop.py --snapshots=1 --epoch_samples_min=500 --minibatch_size=16 --minibatch_per_epoch=5000  2> logs/stderr.log
__log "Starting second iteration of model evaluation"
python rlscout/duel_loop.py --batch_size=32 --games=32 --rollouts=500 --raw_rollouts=500 --iterations=1 --random_rollouts=20 2> logs/stderr.log

rows=`sqlite3 db/models_a.db 'select id, evaluation from models;' | tr -d ' \t\n\r' `
expected="1|+2|+"

if [ "$rows" == "$expected" ]; then
    __log "Success, two model snapshots created"
else
    __log "ERROR: $rows vs $expected"
    exit 1
fi

pkill -P $$

## now starting again in db b

__log " === Testing B === "

__log "Starting model and training data servers in the background"
python rlscout/serve_models.py --db=./db/models_b.db &
python rlscout/serve_samples.py --db=./db/samples_b.db &

__log "Starting first iteration of self-play"
python rlscout/selfplay_loop.py --batch_size=64 --games=64 -t 1 --rollouts=500 --random_rollouts=20
__log "Starting first iteration of model training"
python rlscout/train_loop.py --snapshots=1 --epoch_samples_min=500 --minibatch_size=16 --minibatch_per_epoch=5000 2> logs/stderr.log
__log "Starting first iteration of model evaluation"
python rlscout/duel_loop.py --batch_size=32 --games=32 --rollouts=500 --raw_rollouts=500 --iterations=1 --random_rollouts=20 2> logs/stderr.log

__log "Starting second iteration of self-play"
python rlscout/selfplay_loop.py --batch_size=64 --games=64 -t 1 --rollouts=500 --random_rollouts=20  2> logs/stderr.log
__log "Starting second iteration of model training"
python rlscout/train_loop.py --snapshots=1 --epoch_samples_min=500 --minibatch_size=16 --minibatch_per_epoch=5000  2> logs/stderr.log
__log "Starting second iteration of model evaluation"
python rlscout/duel_loop.py --batch_size=32 --games=32 --rollouts=500 --raw_rollouts=500 --iterations=1 --random_rollouts=20 2> logs/stderr.log

rows=`sqlite3 db/models_b.db 'select id, evaluation from models;' | tr -d ' \t\n\r' `
expected="1|+2|+"

if [ "$rows" == "$expected" ]; then
    __log "Success, two model snapshots created"
else
    __log "ERROR: $rows vs $expected"
    exit 1
fi

model_a=`sqlite3 db/models_a.db 'select torch_model from models where id=2;'`
model_b=`sqlite3 db/models_b.db 'select torch_model from models where id=2;'`

if [ "$model_a" == "$model_b" ]; then
    __log "Success, two models are exactly equal"
else
    __log "ERROR: Models differ"
    exit 1
fi