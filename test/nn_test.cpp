#include <gtest/gtest.h>
#include "../src/graph.h"

using Tensor = miniflow::TensorScalar;

TEST(NodeTest, InputNodeTest)
{
    miniflow::Input X(0.2);
    miniflow::DebugNode D(X);

    EXPECT_EQ(X.is_input(), true);
    EXPECT_EQ(X.getValue().value_, 0.2);
    EXPECT_EQ(X.getGradient()[0].value_, 0.);
    EXPECT_EQ(X.inbound_nodes().size(), size_t(0));

    X.forward();
    X.backward();
    X.update(0.1);

    EXPECT_EQ(X.getValue().value_, 0.2);
    EXPECT_EQ(X.getGradient()[0].value_, 1.);
}

TEST(NodeTest, TrainableNodeTest)
{
    miniflow::Trainable W(0.1);
    miniflow::DebugNode D(W);

    EXPECT_EQ(W.is_input(), true);
    EXPECT_EQ(W.getValue().value_, 0.1);
    EXPECT_EQ(W.getGradient()[0].value_, 0.);
    EXPECT_EQ(W.inbound_nodes().size(), size_t(0));

    W.forward();
    W.backward();
    W.update(0.1);

    EXPECT_EQ(W.getValue().value_, 0.);
    EXPECT_EQ(W.getGradient()[0].value_, 1.);
}

TEST(NodeTest, LinearNodeTest)
{
    miniflow::Input X(0.2);
    miniflow::Trainable W(1), b(0.3);
    miniflow::Linear L(X, W, b);
    miniflow::DebugNode D(L);

    EXPECT_EQ(L.is_input(), false);
    EXPECT_EQ(L.getValue().value_, 0.);
    EXPECT_EQ(L.inbound_nodes().size(), size_t(3));

    miniflow::Graph neural_network(D);
    neural_network.SGD(1., 1);

    EXPECT_EQ(L.getValue().value_, 0.5);
    EXPECT_EQ(W.getGradient()[0].value_, 0.2);
    EXPECT_EQ(W.getValue().value_, 0.8);
}

TEST(NodeTest, SigmoidNodeTest)
{
    miniflow::Input X(0.5);
    miniflow::Sigmoid S(X);
    miniflow::DebugNode D(S);

    S.forward();
    S.backward();
    X.backward();

    EXPECT_NEAR(S.getValue().value_, 0.6224, 1e-3);
    EXPECT_NEAR(X.getGradient()[0].value_, 0.235, 1e-3);
}

TEST(NodeTest, MSENodeTest)
{
    miniflow::Input X(0.6224), Y(0.5);
    miniflow::MSE cost(Y, X);

    cost.forward();
    cost.backward();
    X.backward();
    Y.backward();

    EXPECT_NEAR(cost.getValue().value_, 0.015, 1e-3);
    EXPECT_NEAR(X.getGradient()[0].value_, 0.2448, 1e-3);
    EXPECT_NEAR(Y.getGradient()[0].value_, -0.2448, 1e-3);
}

TEST(NodeTest, SGDTest)
{
    /*
			This is the final test method for computational graph on scalars.
			It attempts to fit Trainables W and b to satisfy the expression 0.5 = sigmoid(W * 0.2 + b)
		*/

    miniflow::Scalar learning_rate = 1.;
    int repeats = 100;

    miniflow::Input X(0.2), Y(0.5);
    miniflow::Trainable W(1), b(0.3);
    miniflow::Linear L(X, W, b);
    miniflow::Sigmoid S(L);
    miniflow::MSE cost(Y, S);

    miniflow::Graph neural_network(cost);
    neural_network.SGD(learning_rate, repeats);

    EXPECT_NEAR(cost.getValue().value_, 0., 1e-10);
}

TEST(NodeTest, DeepNetworkTest)
{
    /*
		This is the final test method for computational graph on scalars.
		It attempts to fit Trainables W and b to satisfy the expression 0.5 = sigmoid(W * 0.2 + b)
		*/

    miniflow::Scalar learning_rate = 1.;
    int repeats = 100;

    miniflow::Input X(0.2), Y(0.5);
    //Layer 1
    miniflow::Trainable W1(1.), b1(0.1);
    miniflow::Linear L1(X, W1, b1);
    miniflow::Sigmoid S1(L1);
    //Layer 2
    miniflow::Trainable W2(1.), b2(0.1);
    miniflow::Linear L2(S1, W2, b2);
    miniflow::Sigmoid S2(L2);
    //Layer 1
    miniflow::Trainable W3(1), b3(0.1);
    miniflow::Linear L3(S2, W3, b3);
    miniflow::Sigmoid S3(L3);
    //
    miniflow::MSE cost(Y, S3);

    miniflow::Graph neural_network(cost);
    neural_network.SGD(learning_rate, repeats);

    EXPECT_NEAR(cost.getValue().value_, 0., 1e-10);
}
