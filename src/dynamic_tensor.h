#pragma once

#include "common.h"
#include "scalar_tensor.h"

namespace dynamictensor
{
	using miniflow::iterate;
	using miniflow::Scalar;
	using miniflow::Index;
	using miniflow::EXP;

	template<unsigned rank> struct Shape
	{
		/*
			Represents a shape of Tensor.
		*/

		Index idx_[rank];
		static constexpr unsigned rank_ = rank;
		using SubShape = typename std::conditional<rank == 1, int, Shape<rank-1>>::type;

		// Shape convolution returns rank - 1 fold of
		// original shape without k's dimention.
		SubShape convoluteShape(int k) const
		{
			if constexpr (rank == 1) return 0;
			else
			{
				SubShape subshape;
				for (int i = 0; i < rank - 1; i++)
				{
					int j = (i < k) ? i : i + 1;
					subshape[i] = idx_[j];
				}
				return subshape;
			}			
		}

		// Folds first dimention.
		// This is the shape of subtensor of 
		// a tensor of given shape.
		SubShape subShape() const
		{
			return convoluteShape(0);
		}

		// Folds last dimention.
		SubShape foldShape() const
		{
			return convoluteShape(rank);
		}

		//Transpose shape
		Shape transpose()
		{
			Shape transposed_shape = *this;
			std::swap(transposed_shape.idx_[rank - 1], transposed_shape.idx_[rank - 2]);
			return transposed_shape;
		}

		// Auxiliary operators

		Index& operator[](int i)
		{
			return idx_[i];
		}

		Index operator[](int i) const
		{
			return idx_[i];
		}

		bool operator==(Shape const& s) const
		{
			for (int i = 0; i < rank; i++)
			{
				if (s.idx_[i] != idx_[i]) return false;
			}
			return true;
		}

		bool operator!=(Shape const& s) const
		{
			return !(operator==(s));
		}
	};

	template<class T, unsigned rank> 
	class Tensor
	{
		/*
			Represents an n-dimensional array of values of a given rank.
			Stored as a std::vector of vectors with a dynamic shape allocated in runtime.
			Only rank of a tensor is static.
		*/

	public:

		// Tempalte statics
		static constexpr unsigned rank_ = rank;
		static constexpr bool is_vector_ = rank_ == 1;
		static constexpr bool is_matrix_ = rank_ == 2;
		using SubTensor = typename std::conditional<is_vector_, T, Tensor<T, rank - 1>>::type;

		std::vector<SubTensor> data_; // main memory structure
		Shape<rank> shape_; // shape of tensor

		// Internal logic

		//
		template<typename F>
		void each(F fn)
		{
			iterate(Index(0), shape_[0], [&](Index i)
			{
				fn(i, data_[i]);
			});
		}

		//
		template<typename F>
		void each(F fn) const
		{
			for (Index i = 0; i < shape_[0]; i++)
			{
				fn(i, data_[i]);
			}
		}
		
		//
		template<typename F>
		Tensor map(F fn) const
		{
			Tensor r(shape_);
			r.each([&](int i, SubTensor& x) {x = fn(data_[i]); });
			return r;
		}

		//
		template<typename F>
		Tensor map_all(F fn) const
		{
			// This is a workaround for the inability to partial specialize member functions.
			// Note that if statement contains a compile-time constant.
			// Only correct branch will be included in the final executable code
			if constexpr(is_vector_)
			{
				static_assert(std::is_same<SubTensor, T>(), "Error in vector branching");
				return map([&](T const& x) {return fn(x); });
			}
			else
			{
				return map([&](SubTensor const& subTensor) {return subTensor.map_all(fn); });
			}
		}

		//
		template<typename F>
		static Tensor zip(Tensor const& t1, Tensor const& t2, F fn)
		{
			assert(t1.shape_ == t2.shape_);
			Tensor r(t1.shape_);
			r.each([&](int i, SubTensor& x) { x = fn(t1[i], t2[i]); });
			return r;
		}
		
		//
		//template<class T, typename F, unsigned rank>
		//static Tensor<T, rank> fold(const Tensor<T, rank>& t, F fn) {}
			
	public:

		// Initialize empty tensor
		Tensor() : Tensor(Shape<rank>()) {}

		// Initialize tensor of given shape
		Tensor(Shape<rank> const& shape) : Tensor(shape, 0) {}

		// Initialize tensor of given shape filled with value
		template<unsigned rank>
		Tensor(Shape<rank> const& shape, T value) : shape_(shape)
		{
			data_.resize(shape[0]);
			for (auto& subTensor : data_)
			{
				subTensor = SubTensor(shape.subShape(), value);
			}
		}

		// Initialize vector of given shape filled with value
		template<>
		Tensor(Shape<1> const& shape, T value) : shape_(shape)
		{
			data_.resize(shape[0], value);
		}

		// Access operator const
		SubTensor operator[](Index i) const
		{
			return data_[i];
		}

		// Access operator nonconst
		SubTensor& operator[](Index i)
		{
			return data_[i];
		}

		// Get shape
		Shape<rank> shape() const
		{
			return shape_;
		}

		// Prints tensor in console		
		void print(int level = 0) const
		{
			for (int i = 0; i <= level; i++) std::cout << " ";
			std::cout << "{";

			if constexpr(is_vector_) // prints vector in console
			{
				for (int i = 0; i < data_.size() - 1; i++)
				{
					std::cout << data_[i] << ", ";
				}
				std::cout << data_.back();
			}
			else // general print branch for non-vector tensor
			{
				std::cout << '\n';
				for (int i = 0; i < data_.size() - 1; i++)
				{
					data_[i].print(level + 1);
					std::cout << "," << '\n';
				}
				data_.back().print(level + 1);
				std::cout << '\n';
				for (int i = 0; i <= level; i++) std::cout << " ";
			}

			std::cout << " }";
			if (level == 0) std::cin.get();
		}

		// Element-wise tensor operations

		friend Tensor operator+(Tensor const& t1, Tensor const& t2)
		{
			return zip(t1, t2, [](SubTensor x1, SubTensor x2) {return x1 + x2; });
		}

		friend Tensor operator-(Tensor const& t1, Tensor const& t2)
		{
			return zip(t1, t2, [](SubTensor x1, SubTensor x2) {return x1 - x2; });
		}

		friend Tensor operator*(Tensor const& t1, Tensor const& t2)
		{
			return zip(t1, t2, [](SubTensor x1, SubTensor x2) {return x1 * x2; });
		}

		friend Tensor operator/(Tensor const& t1, Tensor const& t2)
		{
			return zip(t1, t2, [](SubTensor x1, SubTensor x2) {return x1 / x2; });
		}

		void operator+=(Tensor const& t)
		{
			*this = *this + t;
		}

		void operator-=(Tensor const& t)
		{
			*this = *this - t;
		}

		void operator*=(Tensor const& t)
		{
			*this = *this * t;
		}

		void operator/=(Tensor const& t)
		{
			*this = *this / t;
		}

		// Element-wise operations with scalars

		friend Tensor operator+(Tensor const& t, T s)
		{
			return map(t, [&](SubTensor x) {return x + s; });
		}

		friend Tensor operator+(T s, Tensor const& t)
		{
			return t + s;
		}

		friend Tensor operator-(Tensor const& t, T s)
		{
			return t.map([&](SubTensor x) {return x - s; });
		}

		friend Tensor operator-(T s, Tensor const& t)
		{
			return -1 * t + s;
		}

		friend Tensor operator*(Tensor const& t, T s)
		{
			return t.map([&](SubTensor x) {return x * s; });
		}

		friend Tensor operator*(T s, Tensor const& t)
		{
			return t * s;
		}

		friend Tensor operator/(T s, Tensor const& t)
		{
			return t.map([&](SubTensor x) {return s / x; });
		}

		friend Tensor operator/(Tensor const& t, T s)
		{
			return t * 1 / s;
		}

		friend Tensor operator-(Tensor const& t)
		{
			return t * -1;
		}

		// Math functions

		friend Tensor exp(Tensor const& input)
		{
			return input.map_all([&](T x) {return std::pow(EXP, x); });
		}

		friend Tensor sqr(Tensor const& input)
		{
			return input * input;
		}

		// Special functions

		friend Tensor transpose(Tensor const& input)
		{
			if constexpr(input.is_vector_) return input;
			Tensor transposed(input.shape().transpose());
			if constexpr(input.is_matrix_)
			{
				input.each([&](int i, Tensor<T, 1> const& subtensor)
				{
					subtensor.each([&](int j, T const& x)
					{
						transposed[j][i] = x;
					});
				});
			}
			else
			{
				transposed.each([&](int i, SubTensor& x) 
				{
					x = transpose(input.data_[i]);
				});
			}
			return transposed;
		}

		friend SubTensor sum(Tensor const& input)
		{
			SubTensor sumTensor(input.shape().foldShape());
			if constexpr(input.is_vector_)
			{	
				sumTensor = std::accumulate(input.data_.begin(), input.data_.end(), T(0));
			}
			else
			{
				sumTensor.each([&](int i, typename SubTensor::SubTensor& subTensor) 
				{
					subTensor = sum(input[i]); 
				});
			}
			return sumTensor;
		}

		friend SubTensor mean(Tensor const& input)
		{			
			if constexpr(input.is_vector_)
			{
				return sum(input) / input.shape()[0];
			}
			else
			{
				SubTensor meanTensor(input.shape().foldShape());
				meanTensor.each([&](int i, typename SubTensor::SubTensor& subTensor) 
				{
					subTensor = mean(input[i]); 
				});
				return meanTensor;
			}			
		}

		// Dot

		using DotType = typename std::conditional<rank == 1, T, Tensor>::type;

		friend DotType dot(const Tensor& t1, const Tensor& t2)
		{
			if constexpr(t1.is_vector_)
			{
				assert(t1.shape() == t2.shape());
				return sum(t1*t2);
			}
			else
			{
				assert(t1.shape()[1] == t2.shape()[0]);
				Tensor<T, 2> t2transposed = transpose(t2);
				Tensor<T, 2> result({ t1.shape()[0], t2.shape()[1] });
				t1.each([&](int i, Tensor<T, 1> const& row) {result[i] = dot(t2transposed, row); });
				return result;
			}			
		}

		friend Tensor dot(const Tensor<T, rank + 1>& t1, const Tensor& t2)
		{
			assert(t1.shape()[1] == t2.shape()[0]);
			Tensor<T, 1> result({ t1.shape()[0] });
			t1.each([&](int i, Tensor const& row) {result[i] = dot(row, t2); });
			return result;
		}
	};
} //namespace dynamictensor