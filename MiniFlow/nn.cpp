# include "Graph.h"

void basicNN()
{
	using Tensor = miniflow::TensorScalar;

	miniflow::Scalar learning_rate = 0.1;
	int repeats = 100;

	miniflow::Input<Tensor> X(0.2), Y(0.5);
	miniflow::Trainable<Tensor> W(1), b(0.3);
	miniflow::Linear<Tensor> L(X, W, b);
	miniflow::Sigmoid<Tensor> S(L);
	miniflow::MSE<Tensor> cost(Y, S);

	miniflow::Graph neural_network(cost);
	neural_network.SGD(learning_rate, repeats);

	return;
}

int main()
{
	basicNN();
	
	return 0;
}