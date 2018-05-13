#pragma once

#include <cassert>
#include <iostream>
#include <vector>
#include <initializer_list>
#include <cmath>

namespace dynamictensor
{
	typedef std::size_t Index;
	typedef double Scalar;
	Scalar EXP = 2.71828182845904523536;

	template<unsigned dim> struct Shape
	{
		/*
			Represents a shape of Tensor.
		*/

		Index idx_[dim];
		static const unsigned dim_ = dim;

		// Shape convolution returns dim - 1 fold of
		// original shape without k's dimention.
		Shape<dim - 1> convolutionShape(int k) const
		{
			Shape<dim - 1> subshape;
			for (int i = 0; i < dim - 1; i++)
			{
				int j = (i < k) ? i : i + 1;
				subshape[i] = idx_[j];
			}
			return subshape;
		}

		// Folds first dimention.
		// This is the shape of subtensor of 
		// a tensor of given shape.
		Shape<dim - 1> subShape() const
		{
			return convolutionShape(0);
		}

		// Folds last dimention.
		Shape<dim - 1> foldShape() const
		{
			return convolutionShape(dim);
		} //this name is not cool, come up with the new one

		//Transpose shape
		Shape<2> transpose()
		{
			return Shape<2>{ idx_[1], idx_[0] };
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
			for (int i = 0; i < dim; i++)
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

	template<class T, unsigned dim> class Tensor
	{
		/*
			Represents an n-dimensional array of values.
			Stored as std::vector of vectors,
			whose shape is dynamic and allocated in realtime.
			Only dimention of tensor is static.
		*/

	public:

		// Tempalte statics
		static const unsigned dim_ = dim;
		static const bool is_vector_ = dim_ == 1;
		using SubTensor = typename std::conditional<is_vector_, T, Tensor<T, dim - 1>>::type;

	private:

		std::vector<SubTensor> data_; // main memory structure
		Shape<dim> shape_; // shape of tensor

	public:

		// Internal logic

		//
		template<typename F>
		void each(F fn)
		{
			for (int i = 0; i < shape_[0]; i++)
			{
				fn(i, data_[i]);
			}
		}

		//
		template<typename F>
		void each(F fn) const
		{
			for (int i = 0; i < shape_[0]; i++)
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
			// note that if statement contains a compile-time constant.
			// only correct branch will be included in the final executable code
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

		
		//template<class T, typename F, unsigned dim>
		//static Tensor<T, dim> fold(const Tensor<T, dim>& t, F fn) {}

	public:

		// Initialize empty tensor
		Tensor() : Tensor(Shape<dim>()) {}

		// Initialize tensor of given shape
		Tensor(Shape<dim> const& shape) : Tensor(shape, 0) {}

		// Initialize tensor of given shape filled with value
		template<unsigned dim>
		Tensor(Shape<dim> const& shape, T value) : shape_(shape)
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
		SubTensor operator [](Index i) const
		{
			return data_[i];
		}

		// Access operator nonconst
		SubTensor& operator [](Index i)
		{
			return data_[i];
		}

		// Get shape
		Shape<dim> shape() const
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
				std::cout << std::endl;
				for (int i = 0; i < data_.size() - 1; i++)
				{
					data_[i].print(level + 1);
					std::cout << "," << std::endl;
				}
				data_.back().print(level + 1);
				std::cout << std::endl;
				for (int i = 0; i <= level; i++) std::cout << " ";
			}		

			std::cout << " }";
			if(level == 0) std::cin.get();
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

		friend Tensor operator+(Tensor const& t, Scalar s)
		{
			return map(t, [&](SubTensor x) {return x + s; });
		}

		friend Tensor operator+(Scalar s, Tensor const& t)
		{
			return t + s;
		}

		friend Tensor operator-(Tensor const& t, Scalar s)
		{
			return t.map([&](SubTensor x) {return x - s; });
		}

		friend Tensor operator-(Scalar s, Tensor const& t)
		{
			return -1 * t + s;
		}

		friend Tensor operator*(Tensor const& t, Scalar s)
		{
			return t.map([&](SubTensor x) {return x * s; });
		}

		friend Tensor operator*(Scalar s, Tensor const& t)
		{
			return t * s;
		}

		friend Tensor operator/(Scalar s, Tensor const& t)
		{
			return t.map([&](SubTensor x) {return s / x; });
		}

		friend Tensor operator/(Tensor const& t, Scalar s)
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
			return input.map_all([&](Scalar x) {return std::pow(EXP, x); });
		}

		friend Tensor sqr(Tensor const& input)
		{
			return input * input;
		}
		
	};

	// Special functions

	//Transpose block
	template<class T, unsigned dim>
	Tensor<T, dim> transpose(Tensor<T, dim> const& input)
	{
		Tensor<T, 2> transposed(input.shape());
		transposed.each([&](int i, Tensor<T, dim - 1>& x) {x = transpose(t[i]); });
		return transposed;
	}

	template<class T>
	Tensor<T, 2> transpose(Tensor<T, 2> const& input)
	{
		Tensor<T, 2> transposed(input.shape().transpose());
		input.each([&](int i, Tensor<T, 1> const& subtensor)
		{
			subtensor.each([&](int j, T const& x)
			{
				transposed[j][i] = x;
			});
		});

		return transposed;
	}

	template<class T>
	Tensor<T, 1> transpose(Tensor<T, 1> const& input)
	{
		return input;
	}
	
	//Sum block
	template<class T, unsigned dim>
	Tensor<T, dim-1> sum(Tensor<T, dim> const& input)
	{
		Tensor<T, dim - 1> sumTensor(input.shape().foldShape());
		sumTensor.each([&](int i, typename Tensor<T, dim - 1>::SubTensor& subTensor) {subTensor = sum(input[i]); });
		return sumTensor;
	}

	template<class T>
	T sum(Tensor<T, 1> const& input)
	{
		T sum = 0;
		input.each([&](int i, T const& x) { sum += x; });
		return sum;
	}

	//Mean block
	template<class T, unsigned dim>
	Tensor<T, dim - 1> mean(Tensor<T, dim> const& input)
	{
		Tensor<T, dim - 1> meanTensor(input.shape().foldShape());
		meanTensor.each([&](int i, typename Tensor<T, dim - 1>::SubTensor& subTensor) {subTensor = mean(input[i]); });
		return meanTensor;
	}

	template<class T>
	T mean(Tensor<T, 1> const& input)
	{
		return sum(input) / input.shape()[0];
	}

	//Dot block
	template<class T>
	Tensor<T, 2> dot(const Tensor<T, 2>& t1, const Tensor<T, 2>& t2)
	{
		assert(t1.shape()[1] == t2.shape()[0]);
		Tensor<T, 2> t2transposed = transpose(t2);
		Tensor<T, 2> result({t1.shape()[0], t2.shape()[1]});
		t1.each([&](int i, Tensor<T, 1> const& row) {result[i] = dot(t2transposed, row); });
		return result;
	}

	template<class T>
	Tensor<T, 1> dot(const Tensor<T, 2>& t1, const Tensor<T, 1>& t2)
	{
		assert(t1.shape()[1] == t2.shape()[0]);
		Tensor<T, 1> result({t1.shape()[0]});
		t1.each([&](int i, Tensor<T, 1> const& row) {result[i] = dot(row, t2); });
		return result;
	}

	template<class T>
	T dot(const Tensor<T, 1>& t1, const Tensor<T, 1>& t2)
	{
		assert(t1.shape() == t2.shape());
		return sum(t1*t2);
	}	
}