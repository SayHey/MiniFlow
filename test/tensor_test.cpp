#include <gtest/gtest.h>
#include "../src/tensor.h"
#include "../src/tensor_dynamic.h"
#include "../src/tensor_static.h"

TEST(TensorTest, Basic)
{
    Tensor<double, 1, 2, 3> tensor;
    EXPECT_EQ(tensor.is_matrix_, false);
    EXPECT_EQ(tensor.is_vector_, false);
    EXPECT_EQ(tensor.rank_, 3);
    EXPECT_EQ(tensor.dim_, 1);

    //auto a = tensor.data_[1];
}

TEST(TensorTest, TensorArithmetics)
{
    dynamictensor::Shape<2> shape{2, 5};
    dynamictensor::Tensor<int, 2> tensor1(shape, 2);
    dynamictensor::Tensor<int, 2> tensor2(shape, 3);
    tensor1[0][0] = 4;
    tensor2[1][1] = 5;

    dynamictensor::Tensor<int, 2> tensorPlus = tensor2 + tensor1;  //
    dynamictensor::Tensor<int, 2> tensorMinus = tensor2 - tensor1; //
    dynamictensor::Tensor<int, 2> tensorMult = tensor2 * tensor1;  //
    dynamictensor::Tensor<int, 2> tensorDiv = tensor2 / tensor1;   //

    EXPECT_EQ(tensorPlus[0][0], 7);
    EXPECT_EQ(tensorMinus[0][0], -1);
    EXPECT_EQ(tensorMult[0][0], 12);
}

TEST(TensorTest, TensorMath)
{
    dynamictensor::Shape<2> shape{2, 5};
    dynamictensor::Tensor<double, 2> tensor1(shape, 2.);
    tensor1[0][0] = 3.;

    dynamictensor::Tensor<double, 2> tensorExp = exp(tensor1); //
    dynamictensor::Tensor<double, 2> tensorSQR = sqr(tensor1); //

    EXPECT_NEAR(tensorExp[0][0], 20.0855, 1e-4);
    EXPECT_NEAR(tensorSQR[0][0], 9., 1e-10);
}

TEST(TensorTest, MeanTest)
{
    dynamictensor::Shape<2> shape{2, 5};
    dynamictensor::Tensor<int, 2> tensor(shape, 2);
    tensor[1][1] = 7;

    dynamictensor::Tensor<int, 1> m = mean(tensor); //

    EXPECT_EQ(m[0], 2);
    EXPECT_EQ(m[1], 3);
}

TEST(TensorTest, Transpose)
{
    dynamictensor::Shape<3> shape{3, 2, 5};
    dynamictensor::Tensor<int, 3> tensor(shape, 2);
    tensor[0][1][3] = 7;

    dynamictensor::Tensor<int, 3> transposed = transpose(tensor); //

    EXPECT_EQ(transposed[0][3][1], 7);
}

TEST(TensorTest, DotTest)
{
    dynamictensor::Shape<1> shape1{2};
    dynamictensor::Shape<2> shape3{3, 2};
    dynamictensor::Shape<2> shape4{2, 3};
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

    EXPECT_EQ(dot1, 22);
    EXPECT_EQ(dot2[0], 10);
    EXPECT_EQ(dot2[1], 12);
    EXPECT_EQ(dot3[0][0], 18);
    EXPECT_EQ(dot3[0][1], 15);
    EXPECT_EQ(dot3[1][0], 16);
    EXPECT_EQ(dot3[1][1], 13);
}

TEST(TensorTest, GeneralContainerTest)
{
    statictensor::Tensor<int, 2, 3, 4> tensor =
        {
            {
                {1, 2, 3, 4},
                {5, 6, 7, 8},
                {9, 10, -1, -2},
            },
            {
                {11, 12, 13, 14},
                {15, 16, 17, 18},
                {19, 20, -11, -12},
            }};

    auto shape = tensor.get_shape();

    EXPECT_EQ(tensor[1][0][2], 13);
    EXPECT_EQ(shape[0], unsigned(2));
    EXPECT_EQ(shape[1], unsigned(3));
    EXPECT_EQ(shape[2], unsigned(4));
}
