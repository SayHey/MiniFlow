#pragma once

#include <cassert>
#include <iostream>
#include <vector>
#include <list>
#include <initializer_list>
#include <cmath>
#include <numeric>
#include <execution>

#include <boost/range/adaptor/reversed.hpp>
//#include <algorithm>

namespace miniflow
{
	// Typedefs:
	typedef double Scalar;
	typedef unsigned Index;

	// math constants
	Scalar EXP = 2.71828182845904523536;

	template<typename Iter, typename F>
	void iterateParallel(Iter begin, Iter end, F fn)
	{
		std::for_each(std::execution::par_unseq, begin, end, [&](auto&& item)
		{
			fn(item);
		});
	}

	template<typename Iter, typename F>
	void iterateSerial(Iter begin, Iter end, F fn)
	{
		for (Iter it = begin; it != end; ++it)
		{
			fn(it);
		}
	}

	template<typename Iter, typename F>
	void iterate(Iter begin, Iter end, F fn)
	{
		iterateSerial(begin, end, fn);
	}
}