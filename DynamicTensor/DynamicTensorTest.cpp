#include "DynamicTensor.h"

int main()
{
	dynamictensor::Shape<1> shape1{ 4 };
	dynamictensor::Tensor<int, 1> tensor1(shape1, 3);

	dynamictensor::Shape<2> shape2{ 4, 5 };
	dynamictensor::Tensor<int, 2> tensor2(shape2, 2);
	tensor2[0][0] = 1;

	dynamictensor::Shape<3> shape3{ 2,3,4 };
	dynamictensor::Tensor<int, 3> tensor3(shape3, 3);

	dynamictensor::Tensor<int, 3> tensor4(shape3, 2);

	auto tensor5 = 3* tensor3;

	auto sum = dynamictensor::sum(tensor2);
	auto mean = dynamictensor::mean(tensor1);
	auto dot = dynamictensor::dot(tensor1, tensor1);

	auto tensorT = dynamictensor::Transpose(tensor2);

	tensor2.print();
	sum.print();

	return 0;
}