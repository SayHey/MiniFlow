#pragma once

#include <utility>
#include <vector>
#include <array>
#include <initializer_list>

namespace statictensor2
{
	/*template<class Subtensor, unsigned dim> class TensorBase
	{
		Subtensor data_[dim];
	};

	template<class Subtensor, unsigned dim>
	using Tensor = TensorBase<Tensor<>, dim>;*/

	//template<class Subtensor, unsigned dim, unsigned ... dims>
	//using Tensor<Subtensor> = TensorImpl<Tensor<Subtensor, ... dims>, dim>;
}