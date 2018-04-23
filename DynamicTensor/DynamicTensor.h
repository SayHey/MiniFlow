#pragma once

#include <cassert>
#include <iostream>
#include <vector>
#include <initializer_list>

namespace dynamictensor
{
	typedef std::size_t Index;
	typedef double Scalar;

	template<unsigned dim> struct Shape
	{
		/*
			Represents a shape of Tensor.
		*/

		Index idx_[dim];

		static const unsigned dim_ = dim;

		Index operator[](int i) const 
		{ 
			return idx_[i];
		}

		Index& operator[](int i)
		{
			return idx_[i];
		}

		Shape<dim - 1> subShape() const 
		{ 
			Shape<dim - 1> subshape;
			for (int i = 1; i < dim; i++) subshape[i - 1] = idx_[i];
			return subshape;
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

		using SubTensor = typename std::conditional<dim == 1, T, Tensor<T, dim - 1>>::type;
		std::vector<SubTensor> data_; // main memory structure

		Shape<dim> shape_; // shape of tensor

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

		// Prints tensor in console
		void print() const
		{
			print<dim>();
			std::cin.get();
		}

		template<unsigned d>
		void print(int level = 0) const
		{
			for (int i = 0; i <= level; i++) std::cout << " ";
			std::cout << "{" << std::endl;
			for (int i = 0; i < data_.size() - 1; i++)
			{
				data_[i].print<d - 1>(level + 1);
				std::cout << "," << std::endl;
			}
			data_.back().print<d - 1>(level + 1);
			std::cout <<  std::endl;
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
			return Tensor::zip(t1, t2, [](SubTensor x1, SubTensor x2) {return x1 + x2; });
		}

		friend Tensor operator-(Tensor const& t1, Tensor const& t2)
		{
			return Tensor::zip(t1, t2, [](SubTensor x1, SubTensor x2) {return x1 - x2; });
		}

		friend Tensor operator*(Tensor const& t1, Tensor const& t2)
		{
			return Tensor::zip(t1, t2, [](SubTensor x1, SubTensor x2) {return x1 * x2; });
		}

		friend Tensor operator/(Tensor const& t1, Tensor const& t2)
		{
			return Tensor::zip(t1, t2, [](SubTensor x1, SubTensor x2) {return x1 / x2; });
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
			return Tensor::map(t, [&](SubTensor x) {return x + s; });
		}

		friend Tensor operator+(Scalar s, Tensor const& t)
		{
			return t + s;
		}

		friend Tensor operator-(Tensor const& t, Scalar s)
		{
			return Tensor::map(t, [&](SubTensor x) {return x - s; });
		}

		friend Tensor operator-(Scalar s, Tensor const& t)
		{
			return -1 * t + s;
		}

		friend Tensor operator*(Tensor const& t, Scalar s)
		{
			return Tensor::map(t, [&](SubTensor x) {return x * s; });
		}

		friend Tensor operator*(Scalar s, Tensor const& t)
		{
			return t * s;
		}

		friend Tensor operator/(Scalar s, Tensor const& t)
		{
			return Tensor::map(t, [&](SubTensor x) {return s / x; });
		}

		friend Tensor operator/(Tensor const& t, Scalar s)
		{
			return t * 1 / s;
		}

		friend Tensor operator-(Tensor const& t)
		{
			return t * -1;
		}

		//

		static Tensor exp(const Tensor& t)
		{
			return Tensor::map(t, [&](Scalar x) {return pow(EXP, x); });
		}

		static Tensor sqr(const Tensor& t)
		{
			return t * t;
		}

		static Tensor dot(const Tensor& t1, const Tensor& t2)
		{
			assert(t1.shape_.back() == t2.shape_.front());

			return Tensor(); //placeholder
		}

		static Tensor sum(Tensor const& t)
		{
			return Tensor(); //placeholder
		}

		static Tensor mean(Tensor const& t)
		{
			size_t m = 1; //placeholder
			Scalar d = 1. / m;
			return d * sum(t);
		}

		Tensor Transpose() const
		{
			return Tensor(); //placeholder
		}
	};
}