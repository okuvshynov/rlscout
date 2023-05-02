cd rlscout/rlslib && make all && cd ../..

python rlscout/serve_models.py &
python rlscout/serve_samples.py &
python rlscout/selfplay_loop.py &
python rlscout/duel_loop.py &
python rlscout/train_loop.py &
