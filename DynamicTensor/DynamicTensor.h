#pragma once

#include <cassert>
#include <iostream>
#include <vector>
#include <initializer_list>

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
		Shape<dim - 1> lcShape() const
		{
			return convolutionShape(dim);
		} //this name is not cool, come up with the new one

		//Transpose shape
		static Shape<2> transpose(Shape<2> const& shape)
		{
			return Shape<2>{ shape.idx_[1], shape.idx_[0] };
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

		// Tempalte statics
		static const unsigned dim_ = dim;
		using SubTensor = typename std::conditional<dim == 1, T, Tensor<T, dim - 1>>::type;
		using type = T;

		std::vector<SubTensor> data_; // main memory structure
		Shape<dim> shape_; // shape of tensor

	public:

		// Internal logic
		template<typename F>
		void each(F fn)
		{
			for (int i = 0; i < shape_[0]; i++)
			{
				fn(i, data_[i]);
			}
		}

		template<typename F>
		void each(F fn) const
		{
			for (int i = 0; i < shape_[0]; i++)
			{
				fn(i, data_[i]);
			}
		}

		template<typename F>
		static Tensor zip(Tensor const& t1, Tensor const& t2, F fn)
		{
			assert(t1.shape_ == t2.shape_);
			Tensor r(t1.shape_);
			r.each([&](int i, SubTensor& x) { x = fn(t1[i], t2[i]); });
			return r;
		}

		template<typename F>
		static Tensor map(Tensor const& t, F fn)
		{
			Tensor r(t.shape_);
			r.each([&](int i, SubTensor& x) {x = fn(t[i]); });
			return r;
		}

	public:

		// Initialize empty tensor
		Tensor() : Tensor(Shape<dim>()) {}

		// Initialize tensor of given shape
		Tensor(Shape<dim> shape) : Tensor(shape, 0) {}

		// Initialize tensor of given shape filled with value
		template<unsigned dim>
		Tensor(Shape<dim> shape, T value) : shape_(shape)
		{
			data_.resize(shape[0]);
			for (auto& subTensor : data_)
			{
				subTensor = SubTensor(shape.subShape(), value);
			}
		}

		// Initialize vector of given shape filled with value
		template<>
		Tensor(Shape<1> shape, T value) : shape_(shape)
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
		void print() const
		{
			print<dim>();
			std::cin.get();
		}

		template<unsigned dim>
		void print(int level = 0) const
		{
			for (int i = 0; i <= level; i++) std::cout << " ";
			std::cout << "{" << std::endl;
			for (int i = 0; i < data_.size() - 1; i++)
			{
				data_[i].print<dim - 1>(level + 1);
				std::cout << "," << std::endl;
			}
			data_.back().print<dim - 1>(level + 1);
			std::cout << std::endl;
			for (int i = 0; i <= level; i++) std::cout << " ";
			std::cout << "}";
		}

		// Prints vector in console
		template<>
		void print<1>(int level) const
		{
			for (int i = 0; i <= level; i++) std::cout << " ";
			std::cout << "{ ";
			for (int i = 0; i < data_.size() - 1; i++)
			{

				std::cout << data_[i] << ", ";
			}
			std::cout << data_.back();
			std::cout << " }";
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

		void operator+=(const Tensor& t)
		{
			*this = *this + t;
		}

		void operator-=(const Tensor& t)
		{
			*this = *this - t;
		}

		void operator*=(const Tensor& t)
		{
			*this = *this * t;
		}

		void operator/=(const Tensor& t)
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
			return map(t, [&](SubTensor x) {return x - s; });
		}

		friend Tensor operator-(Scalar s, Tensor const& t)
		{
			return -1 * t + s;
		}

		friend Tensor operator*(Tensor const& t, Scalar s)
		{
			return map(t, [&](SubTensor x) {return x * s; });
		}

		friend Tensor operator*(Scalar s, Tensor const& t)
		{
			return t * s;
		}

		friend Tensor operator/(Scalar s, Tensor const& t)
		{
			return map(t, [&](SubTensor x) {return s / x; });
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

		static Tensor exp(const Tensor& t)
		{
			return map(t, [&](Scalar x) {return pow(EXP, x); });
		}

		static Tensor sqr(const Tensor& t)
		{
			return t * t;
		}

		// Special functions

		static T sum(Tensor<T, 1> const& t)
		{
			T sum = 0;
			t.each([&](int i, SubTensor const& x) { sum += x; });
			return sum;
		}

		static T mean(Tensor<T, 1> const& t)
		{
			return sum(t) / t.shape_[0];
		}

		static T dot(const Tensor<T, 1>& t1, const Tensor<T, 1>& t2)
		{
			assert(t1.shape_ == t2.shape_);
			return sum(t1*t2);
		}

		static Tensor<T, 2> Transpose(Tensor<T, 2> const& input)
		{
			Shape<2> transposedShape = Shape<2>::transpose(input.shape_);
			Tensor<T, 2> transposed(transposedShape);
			input.each([&](int i, SubTensor const& subtensor)
			{
				subtensor.each([&](int j, T const& x)
				{
					transposed[j][i] = x;
				});
			});

			return transposed;
		}

		// now make above special functions tensorwide

		//TODO
		//..
	};
}