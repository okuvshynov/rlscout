all:
	mkdir -p _build
	c++ winner_gen.cpp -std=c++2a -Wall -O3 -o _build/winner_gen
	mkdir -p _gen
	./_build/winner_gen 6 6 > _gen/is_winning6x6x5.h
	./_build/winner_gen 7 7 > _gen/is_winning7x7x5.h
	./_build/winner_gen 8 8 > _gen/is_winning8x8x5.h
	mkdir -p _build
	c++ -shared -std=c++2a -o _build/libmcts.so -fPIC batch_mcts.cpp -O3 -Wall
