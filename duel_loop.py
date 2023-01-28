from local_db import LocalDB
import duel
from utils import save_sample, to_coreml
import time

# does evaluation in FIFO order:
# - pick best model from DB
# - pick first not evaluated model from DB
# - plays against
# - logs the result

# can run in parallel with train/selfplay loops, but not in parallel with itself

db = LocalDB('./_out/8x8/test2.db')

# total ngames * 2 will be played with each player being first
ngames = 5

while True:
    model_to_eval = db.get_last_not_evaluated_model()
    if model_to_eval is None:
        print('no model to evaluate, sleep for a minute')
        time.sleep(60)
        continue
    new = duel.CoreMLPlayer(db, model_to_eval[0], temp=4.0, rollouts=1000, board_size=8)

    best_model = db.get_best_model()
    if best_model is None:
        # create pure MCTS player with many rollouts
        old = duel.CoreMLPlayer(db, 0, temp=2.0, rollouts=500000, board_size=8)
    else:
        old = duel.CoreMLPlayer(db, best_model[0], temp=4.0, rollouts=1000, board_size=8)

    print(f'evaluating model snapshot {model_to_eval[0]}')

    scores = {
        'a': 1,
        '.': 0,
        'b': -1,
    }

    score = 0

    for g in range(ngames):
        score += scores[duel.run(db, new, old)]
        print(f'played {2 * g + 1} games, score is {score}')
        score -= scores[duel.run(db, old, new)]
        print(f'played {2 * g + 2} games, score is {score}')

    result = '+' if 100 * score >= 2 * 10 * ngames else '-'
    db.record_evaluation(model_to_eval[0], result)
        

    