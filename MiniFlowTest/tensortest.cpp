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
		
		TEST_METHOD(SumTest)
		{
			dynamictensor::Shape<1> shape1{ 4 };
			dynamictensor::Tensor<int, 1> tensor1(shape1, 3);
			tensor1[0] = 1;
			auto sum = dynamictensor::Tensor<int, 1>::sum(tensor1);

			//Logger::WriteMessage("SumTest");
			Assert::AreEqual(sum, 10);
		}

		TEST_METHOD(MeanTest)
		{
			dynamictensor::Shape<1> shape1{ 4 };
			dynamictensor::Tensor<int, 1> tensor1(shape1, 3);
			tensor1[1] = 7;
			auto mean = dynamictensor::Tensor<int, 1>::mean(tensor1);

			Assert::AreEqual(mean, 4);
		}

	};
}