#pragma once

#include <math.h>
#include <vector>
#include <initializer_list>

#include "Common.h"


namespace miniflow
{
	template<int dimension>
	struct Shape
	{
		static const int dimension_ = dimension;
		Index shape_[dimension_];
		Index flatten_length_;

		template <typename... T>
		Shape(T... dims) :
			shape_{ dims... }
		{
			flatten_length_ = 1;
			for (int i = 0; i < dimension_; i++)
			{
				flatten_length_ *= shape_[i];
			}
		}

		Index &operator[](Index i)
		{
			return shape_[i];
		}

		const Index &operator[](Index i) const
		{
			return shape_[i];
		}
	};

	template<int dimension>
	class Tensor
	{
		/*
			Represents an n-dimensional array of values.
		*/

		//shape of the tensor
		Shape<dimension> shape_;

		//memory structure
		std::vector<Scalar> value_;
		
	public:

		Tensor(Scalar value):
			shape_{Index(1)},
			value_(1, value)
		{}

		Tensor():
			Tensor(0)
		{}
	};
}