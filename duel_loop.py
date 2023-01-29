from game_client import GameClient
import duel
import time

# does evaluation in FIFO order:
# - pick best model from DB
# - pick first not evaluated model from DB
# - plays against
# - logs the result

# can run in parallel with train/selfplay loops, but not in parallel with itself

client = GameClient()

# total ngames * 2 will be played with each player being first
ngames = 5

while True:
    (model_to_eval_id, model_to_eval) = client.get_model_to_eval()
    #print(model_to_eval)
    if model_to_eval is None:
        print('no model to evaluate, sleep for a minute')
        time.sleep(60)
        continue
    new = duel.CoreMLPlayer(model_to_eval, temp=4.0, rollouts=1000, board_size=8)

    (_, best_model) = client.get_best_model()
    if best_model is None:
        # create pure MCTS player with many rollouts
        old = duel.CoreMLPlayer(torch_model=None, temp=2.0, rollouts=500000, board_size=8)
    else:
        old = duel.CoreMLPlayer(best_model, temp=4.0, rollouts=1000, board_size=8)

    print(f'evaluating model snapshot {model_to_eval_id}')

    scores = {
        'a': 1,
        '.': 0,
        'b': -1,
    }

    score = 0

    for g in range(ngames):
        score += scores[duel.run(new, old)]
        print(f'played {2 * g + 1} games, score is {score}')
        score -= scores[duel.run(old, new)]
        print(f'played {2 * g + 2} games, score is {score}')

    result = '+' if 100 * score >= 2 * 10 * ngames else '-'
    client.record_eval(model_to_eval_id, result)
        

    