#pragma once

#include <vector>
#include <initializer_list>

namespace dynamictensor
{
	template <class T, unsigned N> struct Tensor
	{
		std::vector<Tensor<T, N - 1>> data_;
	};

	template <class T> struct Tensor<T, 0>
	{
		T data_;
	};
}