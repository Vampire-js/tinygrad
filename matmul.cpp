#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>

class Tensor
{
public:
    Tensor(
        std::vector<std::size_t> shape,
        std::vector<double> data) : shape_(std::move(shape)), data_(std::move(data))
    {
        std::size_t expected_elements;

        bool has_zero_dims = false;

        for (const std::size_t dimension : shape_)
        {
            if (dimension == 0)
            {
                has_zero_dims = true;
                expected_elements = 0;
                break;
            }
        }

        if (!has_zero_dims)
        {
            expected_elements = 1;
            for (const std::size_t dimension : shape_)
            {
                if (expected_elements > std::numeric_limits<std::size_t>::max() / dimension)
                {
                    throw std::overflow_error("tensor element count overflows size_t");
                }

                expected_elements *= dimension;
            }
        }

        if (expected_elements != data_.size())
        {
            throw std::invalid_argument("tensor shape does not match its data");
        }
    }

    [[nodiscard]] const std::vector<std::size_t> &shape() const noexcept
    {
        return shape_;
    }

    [[nodiscard]] const std::vector<double> &data() const noexcept
    {
        return data_;
    }

    [[nodiscard]] std::size_t rank() const noexcept
    {
        return shape_.size();
    }

    [[nodiscard]] std::size_t numel() const noexcept
    {
        return data_.size();
    }

    [[nodiscard]] std::size_t dimension(const std::size_t axis) const
    {
        if (axis >= rank())
        {
            throw std::out_of_range("tensor axis is outside its rank");
        }
        return shape_[axis];
    }

    double &at(const std::vector<std::size_t> &idx)
    {
        return data_[flat_index(idx)];
    }

    [[nodiscard]] double at(const std::vector<std::size_t> &idx) const
    {
        return data_[flat_index(idx)];
    }

    [[nodiscard]] double sum() const noexcept
    {
        double result = 0.0;
        for (const double value : data_)
        {
            result += value;
        }

        return result;
    }

    void reshape(std::vector<std::size_t> shape)
    {
        std::size_t expected = 1;
        for (int i : shape)
            expected *= i;
        if (expected == numel() && shape.size() == rank())
            shape_ = std::move(shape);
        else
            throw std::invalid_argument("Incorrect shape");
    }
    void transpose()
    {
        std::reverse(shape_.begin(), shape_.end());
    }

    friend Tensor operator*(const Tensor &a, const Tensor &b);
    friend Tensor operator+(const Tensor &a, const Tensor &b);
    friend Tensor operator-(const Tensor &a, const Tensor &b);

private:
    std::vector<std::size_t> shape_;
    std::vector<double> data_;

    [[nodiscard]] std::size_t flat_index(const std::vector<std::size_t> &idx) const
    {
        if (idx.size() != rank())
        {
            throw std::invalid_argument("number of indices must match tensor rank");
        }

        for (std::size_t axis = 0; axis < rank(); ++axis)
        {
            if (idx[axis] >= shape_[axis])
            {
                throw std::out_of_range("tensor index is outside its dimension");
            }
        }

        std::size_t flat_index = 0;
        std::size_t stride = 1;

        for (std::size_t axis = rank(); axis > 0; --axis)
        {
            const std::size_t current_axis = axis - 1;
            flat_index += idx[current_axis] * stride;
            stride *= shape_[current_axis];
        }

        return flat_index;
    }
};


Tensor operator*(const Tensor& a, const Tensor& b){
    if(a.shape() != b.shape()) throw std::invalid_argument("Operands cannot be multiplied element wise");
    std::vector<double> data(a.numel());
    for(int i=0; i<a.numel(); i++) data[i] = a.data_[i]*b.data_[i];
    Tensor out(a.shape(), data);
    return out;
}

Tensor operator+(const Tensor& a, const Tensor& b){
    if(a.shape() != b.shape()) throw std::invalid_argument("Operands cannot be added element wise");
    std::vector<double> data(a.numel());
    for(int i=0; i<a.numel(); i++) data[i] = a.data_[i]+b.data_[i];
    Tensor out(a.shape(), data);
    return out;
}

Tensor operator-(const Tensor& a, const Tensor& b){
    if(a.shape() != b.shape()) throw std::invalid_argument("Operands cannot be subtracted element wise");
    std::vector<double> data(a.numel());
    for(int i=0; i<a.numel(); i++) data[i] = a.data_[i]+b.data_[i];
    Tensor out(a.shape(), data);
    return out;
}


Tensor matmul(const Tensor &a, Tensor b)
{
    if (a.rank() > 2 || b.rank() > 2)
        throw std::invalid_argument("Input operands aren't matrices");
    if (a.shape()[1] != b.shape()[0])
        throw std::invalid_argument("Input tensor shapes cannot be multiplied");
    std::vector<double> data(a.shape()[0] * b.shape()[1], 0);
    std::size_t l = a.shape()[0];
    std::size_t m = a.shape()[1];
    std::size_t n = b.shape()[1];

    // b.transpose();

    auto flatten = [n](std::size_t i, std::size_t j){
        return (i * n + j);
    };

    for (std::size_t i = 0; i < l; i++){
        for (std::size_t j = 0; j < n; j++){
            for (std::size_t k = 0; k < m; k++){
                data[flatten(i, j)] += a.at({i, k}) * b.at({k, j});
            }
        }
    }
    return Tensor({a.shape()[0], b.shape()[1]}, {data});
}

int main() {
const std::size_t l = 4;
const std::size_t m = 1 << 20; // 1,048,576
const std::size_t n = 4;

    std::vector<double> a_data(l * m, 1.0);
    std::vector<double> b_data(m * n, 1.0);

    Tensor a({l, m}, std::move(a_data));
    Tensor b({m, n}, std::move(b_data));

    std::cout << "Starting matmul...\n";

    Tensor result = matmul(a, b);

    std::cout << "Finished.\n";
}