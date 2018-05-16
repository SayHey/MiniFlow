#include "stdafx.h"
#include "CppUnitTest.h"
#include "CppUnitTestAssert.h"

#include "../MiniFlow/DynamicTensor.h"
#include "../MiniFlow/StaticTensor.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

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
		dynamictensor::Tensor<int, 2> tensor1(shape, 2);
		tensor1[0][0] = 3;

		dynamictensor::Tensor<int, 2> tensorExp = exp(tensor1);  //
		dynamictensor::Tensor<int, 2> tensorSQR = sqr(tensor1); //

		Assert::AreEqual(tensorExp[0][0], 20);
		Assert::AreEqual(tensorSQR[0][0], 9);
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
