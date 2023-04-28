cd rlslib && make all && cd ..

python src/serve_models.py &
python src/serve_samples.py &
python selfplay_loop.py &
python duel_loop.py &
python train_loop.py &
