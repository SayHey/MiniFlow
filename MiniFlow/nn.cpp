# include "Graph.h"

int main()
{
	miniflow::Scalar learning_rate = 0.1;
	int repeats = 10;

	miniflow::Input X(0.2), Y(0.5);
	miniflow::Trainable W(0.1), b(0);
	miniflow::Linear L(X, W, b);
	miniflow::Sigmoid S(L);
	miniflow::MSE cost(Y, S);

	miniflow::Graph neural_network(cost);
	neural_network.SGD(learning_rate, repeats);

	int g = 0;
	
	return 0;
}