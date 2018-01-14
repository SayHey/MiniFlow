#include "StaticTensor.h"
#include "StaticTensor2.h"
#include <iostream>

int main()
{
	statictensor::TensorBase<int, 2> a1{ 1, 2 };
	std::cout << a1[1] << std::endl; // 2

	statictensor::TensorBase<int, 2, 3> a2{ { 1, 2, 3 },{ 4, 5, 6 } };
	std::cout << a2[1][2] << std::endl; // 6

	std::cin.ignore();

	return 0;
}