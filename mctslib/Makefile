all:
	mkdir -p _build
	c++ games/mnk_winner_gen.cpp -std=c++2a -Wall -O3 -o _build/mnk_winner_gen
	mkdir -p _gen
	./_build/mnk_winner_gen 6 6 > _gen/mnk_is_winning6x6x5.h
	./_build/mnk_winner_gen 7 7 > _gen/mnk_is_winning7x7x5.h
	./_build/mnk_winner_gen 8 8 > _gen/mnk_is_winning8x8x5.h
	mkdir -p _build
	c++ -shared -std=c++2a -o _build/libmcts.so -fPIC batch_mcts.cpp -O3 -Wall

othello:
	mkdir -p _build
	c++ games/othello_player.cpp -std=c++2a -Wall -O3 -o _build/othello6x6
	_build/othello6x6
