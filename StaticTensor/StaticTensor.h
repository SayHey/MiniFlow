#pragma once

#include <utility>
#include <vector>
#include <array>
#include <initializer_list>

namespace statictensor
{
	template<class T, unsigned ... dims> 
	class TensorBase;

	template<class T, unsigned dim, unsigned ... dims >
	class TensorBase<T, dim, dims...>
	{
		typedef TensorBase<T, dims...> SubTensor;
		SubTensor data_[dim];

	public:

		TensorBase()
		{}
		
		TensorBase(std::initializer_list<SubTensor> const& il)
		{
			std::copy(il.begin(), il.end(), data_);
		}

		SubTensor& operator[](unsigned i) 
		{ 
			return data_[i]; 
		}
	};

	template<class T, unsigned dim>
	class TensorBase<T, dim>
	{
		T data_[dim];

	public:

		TensorBase()
		{}

		TensorBase(std::initializer_list<T> const& il)
		{
			std::copy(il.begin(), il.end(), data_);
		}

		T& operator[](unsigned i) 
		{ 
			return data_[i]; 
		}
	};
}