rm ./db/tmp_models.py
rm ./db/tmp_samples.py

python src/serve_models.py --db=./db/tmp_models.py &
python src/serve_samples.py --db=./db/tmp_samples.py &

python selfplay_loop.py --batch_size=64 --games=256 -t 4 --rollouts=500 --random_rollouts=50
python train_loop.py --snapshots=1 --epoch_samples_min=1000 --minibatch_size=16
python duel_loop.py --batch_size=32 --games=128 --rollouts=500 --raw_rollouts=500 --iterations=1
python selfplay_loop.py --batch_size=64 --games=256 -t 4 --rollouts=500 --random_rollouts=50
python train_loop.py --snapshots=1 --epoch_samples_min=1000 --minibatch_size=16
python duel_loop.py --batch_size=32 --games=128 --rollouts=500 --raw_rollouts=500 --iterations=1

rows=`sqlite3 db/tmp_models.py 'select id, evaluation from models;' | tr -d ' \t\n\r' `
expected="1|+2|+"

if [ "$rows" == "$expected" ]; then
    echo "Success, two model snapshots created"
    exit 0
else
    echo "ERROR: $rows vs $expected"
    exit 1
fi