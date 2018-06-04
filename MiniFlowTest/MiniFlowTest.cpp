#include "stdafx.h"
#include "CppUnitTest.h"
#include "CppUnitTestAssert.h"

#include "../MiniFlow/DynamicTensor.h"
#include "../MiniFlow/StaticTensor.h"
#include "../MiniFlow/Graph.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
constexpr double eps = 1e-10;

TEST_CLASS(BasicNodeTest)
{
	using Tensor = miniflow::TensorScalar;	

public:

	TEST_METHOD(InputNodeTest)
	{
		miniflow::Input<Tensor> X(0.2);
		miniflow::DebugNode<Tensor> D(X);
		
		Assert::AreEqual(X.is_input(), true);
		Assert::AreEqual(X.getValue().value_, 0.2);
		Assert::AreEqual(X.getGradient()[0].value_, 0.);
		Assert::AreEqual(X.inbound_nodes().size(), size_t(0));

		X.forward();
		X.backward();
		X.update(0.1);

		Assert::AreEqual(X.getValue().value_, 0.2);
		Assert::AreEqual(X.getGradient()[0].value_, 1.);
	}

	TEST_METHOD(TrainableNodeTest)
	{
		miniflow::Trainable<Tensor> W(0.1);
		miniflow::DebugNode<Tensor> D(W);

		Assert::AreEqual(W.is_input(), true);
		Assert::AreEqual(W.getValue().value_, 0.1);
		Assert::AreEqual(W.getGradient()[0].value_, 0.);
		Assert::AreEqual(W.inbound_nodes().size(), size_t(0));

		W.forward();
		W.backward();
		W.update(0.1);

		Assert::AreEqual(W.getValue().value_, 0.);
		Assert::AreEqual(W.getGradient()[0].value_, 1.);
	}

	TEST_METHOD(LinearNodeTest)
	{
		miniflow::Input<Tensor> X(0.2);
		miniflow::Trainable<Tensor> W(1), b(0.3);
		miniflow::Linear<Tensor> L(X, W, b);
		miniflow::DebugNode<Tensor> D(L);

		Assert::AreEqual(L.is_input(), false);
		Assert::AreEqual(L.getValue().value_, 0.);
		Assert::AreEqual(L.inbound_nodes().size(), size_t(3));

		miniflow::Graph neural_network(D);
		neural_network.SGD(1., 1);

		Assert::AreEqual(L.getValue().value_, 0.5);
		Assert::AreEqual(W.getGradient()[0].value_, 0.2);
		Assert::AreEqual(W.getValue().value_, 0.8);
		
	}

	TEST_METHOD(SigmoidNodeTest)
	{
		miniflow::Input<Tensor> X(0.5);
		miniflow::Sigmoid<Tensor> S(X);
		miniflow::DebugNode<Tensor> D(S);

		S.forward();
		S.backward();
		X.backward();

		Assert::AreEqual(S.getValue().value_, 0.6224, 1e-3);
		Assert::AreEqual(X.getGradient()[0].value_, 0.235, 1e-3);
	}

	TEST_METHOD(MSENodeTest)
	{
		miniflow::Input<Tensor> X(0.6224), Y(0.5);
		miniflow::MSE<Tensor> cost(Y, X);

		cost.forward();
		cost.backward();
		X.backward();
		Y.backward();

		Assert::AreEqual(cost.getValue().value_, 0.015, 1e-3);
		Assert::AreEqual(X.getGradient()[0].value_, 0.2448, 1e-3);
		Assert::AreEqual(Y.getGradient()[0].value_, -0.2448, 1e-3);
	}

	TEST_METHOD(FullNetworkTest)
	{
		/*
			This is the final test method for computational graph on scalars.
			It attempts to fit Trainables W and b to satisfy the expression 0.5 = sigmoid(W * 0.2 + b)
		*/

		miniflow::Scalar learning_rate = 0.01;
		int repeats = 10;

		miniflow::Input<Tensor> X(0.2), Y(0.5);
		miniflow::Trainable<Tensor> W(1), b(0.3);
		miniflow::Linear<Tensor> L(X, W, b);
		miniflow::Sigmoid<Tensor> S(L);
		miniflow::MSE<Tensor> cost(Y, S);

		miniflow::Graph neural_network(cost);
		neural_network.SGD(learning_rate, repeats);

		Assert::AreEqual(cost.getValue().value_, 0., 1e-3);
	}
};

TEST_CLASS(DynamicTensorTest)
{
public:

	TEST_METHOD(TensorArithmetics)
	{
		dynamictensor::Shape<2> shape{ 2, 5 };
		dynamictensor::Tensor<int, 2> tensor1(shape, 2);
		dynamictensor::Tensor<int, 2> tensor2(shape, 3);
		tensor1[0][0] = 4;
		tensor2[1][1] = 5;

		dynamictensor::Tensor<int, 2> tensorPlus = tensor2 + tensor1;  //
		dynamictensor::Tensor<int, 2> tensorMinus = tensor2 - tensor1; //
		dynamictensor::Tensor<int, 2> tensorMult = tensor2 * tensor1;  //
		dynamictensor::Tensor<int, 2> tensorDiv = tensor2 / tensor1;   //

		Assert::AreEqual(tensorPlus[0][0], 7);
		Assert::AreEqual(tensorMinus[0][0], -1);
		Assert::AreEqual(tensorMult[0][0], 12);
	}

	TEST_METHOD(TensorMath)
	{
		dynamictensor::Shape<2> shape{ 2, 5 };
		dynamictensor::Tensor<double, 2> tensor1(shape, 2.);
		tensor1[0][0] = 3.;

		dynamictensor::Tensor<double, 2> tensorExp = exp(tensor1);  //
		dynamictensor::Tensor<double, 2> tensorSQR = sqr(tensor1); //

		Assert::AreEqual(tensorExp[0][0], 20.0855, 1e-4);
		Assert::AreEqual(tensorSQR[0][0], 9., 1e-10);
	}

	TEST_METHOD(MeanTest)
	{
		dynamictensor::Shape<2> shape{ 2, 5 };
		dynamictensor::Tensor<int, 2> tensor(shape, 2);
		tensor[1][1] = 7;

		dynamictensor::Tensor<int, 1> m = mean(tensor); //

		Assert::AreEqual(m[0], 2);
		Assert::AreEqual(m[1], 3);
	}

	TEST_METHOD(Transpose)
	{
		dynamictensor::Shape<3> shape{ 3, 2, 5 };
		dynamictensor::Tensor<int, 3> tensor(shape, 2);
		tensor[0][1][3] = 7;

		dynamictensor::Tensor<int, 3> transposed = transpose(tensor); //

		Assert::AreEqual(transposed[0][3][1], 7);
	}

	TEST_METHOD(DotTest)
	{
		dynamictensor::Shape<1> shape1{ 2 };
		dynamictensor::Shape<2> shape3{ 3,2 };
		dynamictensor::Shape<2> shape4{ 2,3 };
		dynamictensor::Tensor<int, 1> tensor1(shape1, 2);
		dynamictensor::Tensor<int, 1> tensor2(shape1, 3);
		dynamictensor::Tensor<int, 2> tensor3(shape3, 2);
		dynamictensor::Tensor<int, 2> tensor4(shape4, 3);
		tensor1[0] = 4;
		tensor2[1] = 5;
		tensor3[0][1] = 1;
		tensor4[1][1] = 2;

		int dot1 = dot(tensor1, tensor2);
		dynamictensor::Tensor<int, 1> dot2 = dot(tensor3, tensor1);
		dynamictensor::Tensor<int, 2> dot3 = dot(tensor4, tensor3);

		Assert::AreEqual(dot1, 22);
		Assert::AreEqual(dot2[0], 10);
		Assert::AreEqual(dot2[1], 12);
		Assert::AreEqual(dot3[0][0], 18);
		Assert::AreEqual(dot3[0][1], 15);
		Assert::AreEqual(dot3[1][0], 16);
		Assert::AreEqual(dot3[1][1], 13);
	}
};

TEST_CLASS(StaticTensorTest)
{
public:

	TEST_METHOD(GeneralContainerTest)
	{
		statictensor::Tensor<int, 2, 3, 4> tensor = 
		{ 
			{ 
				{1,2,3,4},
				{5,6,7,8},
				{9,10,-1,-2},
			},
			{ 
				{ 11,12,13,14 },
				{ 15,16,17,18 },
				{ 19,20,-11,-12 },
			} 
		};
		
		auto shape = tensor.get_shape();

		Assert::AreEqual(tensor[1][0][2], 13);
		Assert::AreEqual(shape[0], unsigned(2));
		Assert::AreEqual(shape[1], unsigned(3));
		Assert::AreEqual(shape[2], unsigned(4));
	}

};
