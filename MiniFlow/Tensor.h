#pragma once

#include "Common.h"
#include "../DynamicTensor/DynamicTensor.h"

namespace miniflow
{
	class TensorScalar
	{
		/*
			PLACEHOLDER CLASS for the purpose of debugginc of the computational graph.
			Is basically a Scalar that supports all the functions of generic Tensor.
		*/

		Scalar value_;

	public:

		TensorScalar() :
			value_(0)
		{}

		TensorScalar(Scalar value) :
			value_(value)
		{}

		Scalar operator[](int) const
		{
			return value_;
		}

		Scalar& operator[](int)
		{
			return value_;
		}

		// Element-wise tensor operations

		friend TensorScalar operator+(TensorScalar const& t1, TensorScalar const& t2)
		{
			return t1.value_ + t2.value_;
		}

		friend TensorScalar operator-(TensorScalar const& t1, TensorScalar const& t2)
		{
			return t1.value_ - t2.value_;
		}

		friend TensorScalar operator*(TensorScalar const& t1, TensorScalar const& t2)
		{
			return t1.value_ * t2.value_;
		}

		friend TensorScalar operator/(TensorScalar const& t1, TensorScalar const& t2)
		{
			return t1.value_ / t2.value_;
		}

		void operator+=(const TensorScalar& t)
		{
			*this = *this + t;
		}

		void operator-=(const TensorScalar& t)
		{
			*this = *this - t;
		}

		void operator*=(const TensorScalar& t)
		{
			*this = *this * t;
		}

		void operator/=(const TensorScalar& t)
		{
			*this = *this / t;
		}

		// Element-wise operations with scalars

		friend TensorScalar operator+(TensorScalar const& t, Scalar s)
		{
			return t.value_ + s;
		}

		friend TensorScalar operator+(Scalar s, TensorScalar const& t)
		{
			return t + s;
		}

		friend TensorScalar operator-(TensorScalar const& t, Scalar s)
		{
			return t.value_ - s;
		}

		friend TensorScalar operator-(Scalar s, TensorScalar const& t)
		{
			return s - t.value_;
		}

		friend TensorScalar operator*(TensorScalar const& t, Scalar s)
		{
			return s * t.value_;
		}

		friend TensorScalar operator*(Scalar s, TensorScalar const& t)
		{
			return t * s;
		}

		friend TensorScalar operator/(Scalar s, TensorScalar const& t)
		{
			return s / t.value_;
		}

		friend TensorScalar operator/(TensorScalar const& t, Scalar s)
		{
			return t * 1 / s;
		}

		friend TensorScalar operator-(TensorScalar const& t)
		{
			return t * -1;
		}

		// Special functions

		static TensorScalar sum(TensorScalar const& t)
		{
			return TensorScalar(t.value_);
		}

		static TensorScalar mean(TensorScalar const& t)
		{
			return TensorScalar(t.value_);
		}

		static TensorScalar exp(const TensorScalar& t)
		{
			return pow(EXP, t.value_);
		}

		static TensorScalar sqr(const TensorScalar& t)
		{
			return t * t;
		}

		static TensorScalar dot(const TensorScalar& t1, const TensorScalar& t2)
		{
			return TensorScalar(t1.value_ * t2.value_);
		}

		TensorScalar T() const
		{
			return TensorScalar(value_);
		}
	};
}