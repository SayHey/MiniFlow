#pragma once

#include <utility>
#include <vector>
#include <array>
#include <initializer_list>

namespace statictensor
{
	template<typename TensorContainer>
	class TensorImp
	{
		/*
		Represents an n-dimensional array of values.
		Stored as array of arrays wath a static shape allocated in compiletime.
		*/

		using SubTensor = typename TensorContainer::SubTensor;
		SubTensor data_[TensorContainer::dim_];

	public:

		TensorImp() = default;

		TensorImp(std::initializer_list<SubTensor> const& il)
		{
			std::copy(il.begin(), il.end(), data_);
		}

		SubTensor& operator[](unsigned i)
		{
			return data_[i];
		}
	};

	template<typename T, unsigned ... dims>
	struct TensorContainer;

	template<typename T, unsigned ... dims>
	using Tensor = TensorImp<TensorContainer<T, dims...>>;

	template<typename T, unsigned dim, unsigned ... dims >
	struct TensorContainer<T, dim, dims...>
	{
		static const unsigned dim_ = dim;
		using SubTensor = Tensor<T, dims...>;
		
	};

	template<typename T, unsigned dim>
	struct TensorContainer<T, dim>
	{
		static const unsigned dim_ = dim;
		using SubTensor = T;
	};
}