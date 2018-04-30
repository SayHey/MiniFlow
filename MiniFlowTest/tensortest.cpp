#include "stdafx.h"
#include "CppUnitTest.h"
#include "CppUnitTestAssert.h"

#include "..\DynamicTensor\DynamicTensor.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace MiniFlowTest
{		
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
		
		TEST_METHOD(SumTest)
		{
			dynamictensor::Shape<2> shape{ 2, 5 };
			dynamictensor::Tensor<int, 2> tensor(shape, 2);
			tensor[0][0] = 1;

			dynamictensor::Tensor<int, 1> sum = dynamictensor::sum(tensor); //

			Assert::AreEqual(sum[0], 9);
			Assert::AreEqual(sum[1], 10);
		}

		TEST_METHOD(MeanTest)
		{
			dynamictensor::Shape<2> shape{ 2, 5 };
			dynamictensor::Tensor<int, 2> tensor(shape, 2);
			tensor[1][1] = 7;

			dynamictensor::Tensor<int, 1> mean = dynamictensor::mean(tensor); //

			Assert::AreEqual(mean[0], 2);
			Assert::AreEqual(mean[1], 3);
		}

	};
}