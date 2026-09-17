#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include "tensor.h"

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