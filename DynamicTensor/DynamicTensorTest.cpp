#include "DynamicTensor.h"

int main()
{
	dynamictensor::Shape<1> shape1{ 5 };
	dynamictensor::Shape<2> shape2{ 4, 5 };
	dynamictensor::Shape<3> shape3{ 2,3,4 };
	dynamictensor::Tensor<double, 1> tensor1(shape1, 3);
	dynamictensor::Tensor<double, 2> tensor2(shape2, 2);
	dynamictensor::Tensor<double, 3> tensor3(shape3, 3);
	dynamictensor::Tensor<double, 3> tensor4(shape3, 2);

	auto s = sum(tensor2);
	auto m = mean(tensor2);
	auto tensorT = transpose(tensor2);

	auto dot1 = dot(tensor1, tensor1);
	auto dot2 = dot(tensor2, tensor1);
	//auto dot3 = dynamictensor::dot(tensorT, tensor2);

	tensor3.print();

	return 0;
}