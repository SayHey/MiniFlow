#pragma once

#include "Common.h"
#include <math.h>

namespace miniflow
{
	class Tensor
	{
		/*
			Represents an n-dimensional array of values.
		*/	

		// TODO: Implement
		// 
		Scalar value_; //placeholder

		template<typename F>
		void each(F fn)
		{
			fn(0, value_); //placeholder

			//for (int i = 0; i < dim; i++)
			//{
				//fn(i, _a[i]);
			//}
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

		Tensor():
			value_(0)
		{}

		Tensor(Scalar value):
			value_(value)
		{}

		Scalar operator[](int i) const
		{
			return value_; //Placeholder;
		}

		Scalar& operator[](int i)
		{
			return value_; //Placeholder;
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

		static Tensor sum(Tensor const& t)
		{
			return Tensor(t.value_); //placeholder
		}

		static Tensor mean(Tensor const& t)
		{
			std::size_t m = 1; //placeholder
			Scalar d = 1. / m;
			return d * sum(t);
		}		

		static Tensor exp(const Tensor& t)
		{
			return Tensor::map(t, [&](Scalar x) {return pow(EXP, x); });
		}

		static Tensor sqr(const Tensor& t)
		{
			return t*t;
		}

		static Tensor dot(const Tensor& t1, const Tensor& t2)
		{
			return Tensor(t1.value_ * t2.value_); //placeholder
		}

		Tensor T() const
		{
			return *this; //placeholder
		}
	};
}