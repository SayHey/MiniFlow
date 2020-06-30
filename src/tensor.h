/*
    A Tensor class:
    1) incapsulates multidimentional array
    2) 

*/

#include <array>
#include <tuple>
#include <type_traits>

using Index = unsigned;

struct TensorInterface
{

};

template<typename T, unsigned ... dims>
struct Tensor;

template <typename T, Index dim, Index ... dims>
struct Tensor<T, dim, dims...> : public TensorInterface
{
    static constexpr Index dim_ = dim;
    static constexpr Index rank_ = sizeof...(dims) + 1;
    static constexpr bool is_vector_ = rank_ == 1;
	static constexpr bool is_matrix_ = rank_ == 2;

    using Shape = std::array<Index, rank_>;
    static constexpr Shape shape_ = {dim, dims...};

    using SubTensor = typename std::conditional<is_vector_, T, Tensor<T, dims...>>::type;
    std::array<SubTensor, dim_> data_;
};