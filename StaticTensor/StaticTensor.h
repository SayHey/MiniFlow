#pragma once

#include <utility>
#include <vector>
#include <array>
#include <initializer_list>

namespace statictensor
{
	template<unsigned rank>
	using Shape = std::array<unsigned, rank>;

	template<typename TensorContainer>
	class TensorImp
	{
		/*
		Represents an n-dimensional array of values.
		Stored as array of arrays wath a static shape allocated in compiletime.
		*/

		using SubTensor = typename TensorContainer::SubTensor;
		static constexpr unsigned dim_ = TensorContainer::dim_;
		static constexpr unsigned rank_ = TensorContainer::rank_;
		//static constexpr Shape<rank_> shape_ = get_shape_static();

		SubTensor data_[dim_];

		static constexpr Shape<rank_> get_shape_static()
		{
			if constexpr(rank_ == 1)
			{
				return { dim_ };
			}
			else
			{
				Shape<rank_> shape;
				Shape<rank_ - 1> subTensor_shape = SubTensor::get_shape_static();
				std::copy(subTensor_shape.begin(), subTensor_shape.end(), shape.begin() + 1);
				shape[0] = dim_;
				return shape;
			}
		}

		template<typename C>
		friend class TensorImp;
		
	public:

		constexpr TensorImp() = default;

		constexpr TensorImp(std::initializer_list<SubTensor> const& il)
		{
			std::copy(il.begin(), il.end(), data_);
		}

		constexpr SubTensor& operator[](unsigned i)
		{
			return data_[i];
		}

		constexpr SubTensor operator[](unsigned i) const
		{
			return data_[i];
		}
		
		constexpr Shape<rank_> get_shape() const
		{
			return TensorImp::get_shape_static();
		}		
	};

	template<typename T, unsigned ... dims>
	struct TensorContainer;

	template<typename T, unsigned ... dims>
	using Tensor = TensorImp<TensorContainer<T, dims...>>;

	template<typename T, unsigned dim, unsigned ... dims >
	struct TensorContainer<T, dim, dims...>
	{
		using SubTensor = Tensor<T, dims...>;
		static constexpr unsigned dim_ = dim;
		static constexpr unsigned rank_ = sizeof...(dims) + 1;
	};

	template<typename T, unsigned dim>
	struct TensorContainer<T, dim>
	{		
		using SubTensor = T;
		static constexpr unsigned dim_ = dim;
		static constexpr unsigned rank_ = 1;
	};
} //namespace statictensor