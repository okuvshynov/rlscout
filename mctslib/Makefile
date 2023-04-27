all:
	mkdir -p _build
	c++ -shared -std=c++2a -o _build/libmcts.so -fPIC puct/batch_mcts.cpp -O3 -Wall -I.

othello_ab:
	mkdir -p _build
	c++ othello/othello_6x6_ab.cpp -std=c++2a -Wall -O3 -o _build/othello6x6_ab -I.
