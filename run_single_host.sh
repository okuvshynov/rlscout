cd mctslib && make all && cd ..

python src/serve_models.py &
python src/serve_samples.py &
python selfplay_loop.py &
python duel_loop.py &
python train_action_value.py &
