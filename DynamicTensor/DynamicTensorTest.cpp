#include "DynamicTensor.h"

int main()
{
	dynamictensor::Tensor<int, 1> shape1{ { 1, 2 } };
	dynamictensor::Tensor<int, 2> shape2{ { { 1, 2, 3 },{ 4, 5, 6 } } };

	return 0;
}