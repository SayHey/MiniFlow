#include "DynamicTensor.h"

int main()
{
	dynamictensor::Index size = 5;
	dynamictensor::Tensor<int, 1> tensor1(size);

	dynamictensor::Shape<2> shape2{ 4, 5 };
	dynamictensor::Tensor<int, 2> tensor2(shape2, 1);

	dynamictensor::Shape<3> shape3{ 2,3,4 };
	dynamictensor::Tensor<int, 3> tensor3(shape3, 1);

	dynamictensor::Tensor<int, 3> tensor4(shape3);

	tensor4.print();

	//dynamictensor::Tensor<int, 1> tensor1{ 1, 2 };
	//dynamictensor::Tensor<int, 2> tensor2{ { 1, 2, 3 },{ 4, 5, 6 } };

	return 0;
}