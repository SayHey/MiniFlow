#include "DynamicTensor.h"

int main()
{
	dynamictensor::Shape<1> shape1{ 4 };
	dynamictensor::Tensor<double, 1> tensor1(shape1, 3);

	dynamictensor::Shape<2> shape2{ 4, 5 };
	dynamictensor::Tensor<double, 2> tensor2(shape2, 2);
	tensor2[0][0] = 1;

	dynamictensor::Shape<3> shape3{ 2,3,4 };
	dynamictensor::Tensor<double, 3> tensor3(shape3, 3);

	dynamictensor::Tensor<double, 3> tensor4(shape3, 2);

	auto tensor5 = exp(tensor1);

	auto sum = dynamictensor::sum(tensor2);
	auto mean = dynamictensor::mean(tensor2);
	auto dot = dynamictensor::dot(tensor1, tensor1);
	auto tensorT = dynamictensor::transpose(tensor2);

	tensor2.print();
	mean.print();

	return 0;
}