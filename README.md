Next:

[ ] refactor to make it easier to implement self-play, full loop with distribution
[ ] full loop 
[ ] abstract away game from search?
[ ] value model
[ ] load to colab and try interactively?
[ ] examples 
    [ ] the one with attack/defence which works for 1M rollouts 
    [ ] how model-only player gets all draws against deeper search, but loses occasionally to more shallow
[ ] visualization for activations/gradients
[ ] understand the ANE performance, implement batching

[x] multitrheading self-play
[x] use model within search
    [x] allow ANE use for this?
[-] avoid sending 'handles' back, just send the state itself for state and buffer/root for mcts?
[x] residual net
[x] experiment on ANN -- just see what the throughput is depending on batch size
[x] experiment on Model only vs MCTS Only vs MCTS Search (no value model here)