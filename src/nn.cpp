# include "graph.h"

void basicNN()
{
	//using Tensor = miniflow::TensorScalar;

	miniflow::Scalar learning_rate = 0.1;
	int repeats = 500;

	miniflow::Input X(0.2), Y(0.5);
	miniflow::Trainable W(1), b(0.3);
	miniflow::Linear L(X, W, b);
	miniflow::Sigmoid S(L);
	miniflow::MSE cost(Y, S);

	miniflow::Graph neural_network(cost);
	neural_network.SGD(learning_rate, repeats);

	return;
}

int main()
{
	basicNN();
	
	return 0;
}