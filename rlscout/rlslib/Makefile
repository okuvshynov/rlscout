all:
	mkdir -p _build
	c++ -shared -std=c++2a -o _build/librls.so -fPIC rlslib.cpp -O3 -Wall -I.
	c++ tests/test_random_action.cpp -std=c++2a -Wall -O3 -o _build/test_random_action -I.
	c++ tests/test_othello_board_ops.cpp -std=c++2a -Wall -O3 -o _build/test_othello_board_ops -I.
	./_build/test_random_action
	./_build/test_othello_board_ops

othello_ab:
	mkdir -p _build
	c++ othello/othello_6x6_ab.cpp -std=c++2a -Wall -O3 -o _build/othello6x6_ab -I.
