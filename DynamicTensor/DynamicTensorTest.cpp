#include "DynamicTensor.h"

int main()
{
	dynamictensor::Shape<1> shape1{ 4 };
	dynamictensor::Tensor<int, 1> tensor1(shape1, 3);

	dynamictensor::Shape<2> shape2{ 4, 5 };
	dynamictensor::Tensor<int, 2> tensor2(shape2, 1);

	dynamictensor::Shape<3> shape3{ 2,3,4 };
	dynamictensor::Tensor<int, 3> tensor3(shape3, 3);

	dynamictensor::Tensor<int, 3> tensor4(shape3, 2);

	auto tensor5 = 3* tensor3;

	auto sum = dynamictensor::Tensor<int, 1>::sum(tensor1);
	auto mean = dynamictensor::Tensor<int, 1>::mean(tensor1);
	auto dot = dynamictensor::Tensor<int, 1>::dot(tensor1, tensor1);

	auto tensorT = dynamictensor::Tensor<int, 2>::Transpose(tensor2);

	tensor5.print();

	return 0;
}