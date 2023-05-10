#!/bin/bash

__cleanup ()
{
    pkill -P $$
}
trap __cleanup EXIT

echo "Starting model and training data servers in the background"
python rlscout/serve_models.py --db=./db/othello6x6_models2a.db &
python rlscout/serve_samples.py --db=./db/othello6x6_samples2a.db &

while true
do
    echo "Waiting for status update"
    sleep 30
    echo "Current model db status:"
    sqlite3 db/othello6x6_models2a.db 'select id, evaluation from models  order by id desc limit 10;'
    echo "Current sample db status:"
    sqlite3 db/othello6x6_samples2a.db 'select count(*) from samples;'
done
