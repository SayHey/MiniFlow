#pragma once

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

		Index operator [](int i) const 
		{ 
			return idx_[i];
		}

		Index& operator [](int i)
		{
			return idx_[i];
		}

		Shape<dim - 1> subShape() const 
		{ 
			Shape<dim - 1> subshape;
			for (int i = 1; i < dim; i++) subshape[i - 1] = idx_[i];
			return subshape;
		}
	};

	template<class T, unsigned dim> class Tensor
	{
		/*
			Represents an n-dimensional array of values.
			Stored as std::vector of vectors, 
			whose size is dynamic and allocated in realtime.
		*/

		using SubTensor = typename std::conditional<dim == 1, T, Tensor<T, dim - 1>>::type;
		//typedef std::conditional<dim == 1, T, Tensor<T, dim - 1>>::type SubTensor;
		std::vector<SubTensor> data_; // Main memory structure


		// Internal logic
		template<typename F>
		void each(F fn)
		{
			for (int i = 0; i < dim; i++)
			{
				fn(i, data_[i]);
			}
		}

		template<typename F>
		static Tensor zip(Tensor const& v1, Tensor const& v2, F fn)
		{
			Tensor r;
			r.each([&](int i, Scalar& x) {x = fn(v1[i], v2[i]); });
			return r;
		}

		template<typename F>
		static Tensor map(Tensor const& v, F fn)
		{
			Tensor r;
			r.each([&](int i, Scalar& x) {x = fn(v[i]); });
			return r;
		}

	public:

		// Initialize empty tensor
		Tensor() : Tensor(0) {}

		// Initialize tensor of given size
		Tensor(Index size)
		{
			data_.resize(size);
		}

		Tensor(Shape<dim> shape) : Tensor(shape, 0) {}

		// Initialize tensor of given shape filled with value
		template<unsigned dim>
		Tensor(Shape<dim> shape, T value)
		{
			data_.resize(shape[0]);
			for (auto& subTensor : data_)
			{
				subTensor = SubTensor(shape.subShape(), value);
			}
		}

		// Initialize vector of given shape filled with value
		template<>
		Tensor(Shape<1> shape, T value)
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
			return Tensor::zip(t1, t2, [](Scalar x1, Scalar x2) {return x1 + x2; });
		}

		friend Tensor operator-(Tensor const& t1, Tensor const& t2)
		{
			return Tensor::zip(t1, t2, [](Scalar x1, Scalar x2) {return x1 - x2; });
		}

		friend Tensor operator*(Tensor const& t1, Tensor const& t2)
		{
			return Tensor::zip(t1, t2, [](Scalar x1, Scalar x2) {return x1 * x2; });
		}

		friend Tensor operator/(Tensor const& t1, Tensor const& t2)
		{
			return Tensor::zip(t1, t2, [](Scalar x1, Scalar x2) {return x1 / x2; });
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
			return Tensor::map(t, [&](Scalar x) {return x + s; });
		}

		friend Tensor operator+(Scalar s, Tensor const& t)
		{
			return t + s;
		}

		friend Tensor operator-(Tensor const& t, Scalar s)
		{
			return Tensor::map(t, [&](Scalar x) {return x - s; });
		}

		friend Tensor operator-(Scalar s, Tensor const& t)
		{
			return t - s;
		}

		friend Tensor operator*(Tensor const& t, Scalar s)
		{
			return Tensor::map(t, [&](Scalar x) {return x * s; });
		}

		friend Tensor operator*(Scalar s, Tensor const& t)
		{
			return t * s;
		}

		friend Tensor operator/(Scalar s, Tensor const& t)
		{
			return Tensor::map(t, [&](Scalar x) {return s / x; });
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
		/*
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
			return Tensor(); //placeholder
		}

		Tensor T() const
		{
			return *this; //placeholder
		}
		*/
	};
}