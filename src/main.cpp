#include <iostream>
#include "tensor.h"
#include "matmul_gpu.h"

int main()
{
    Tensor A(
        {2, 3},
        {1, 2, 3,
         4, 5, 6});

    Tensor B(
        {3, 2},
        {7, 8,
         9, 10,
         11, 12});
    Tensor C(
        {2, 2},
        {0, 0,
         0, 0});

    matmul_gpu(A, B, C);

    std::cout << C.at({0, 0}) << ' ' << C.at({0, 1}) << '\n';
    std::cout << C.at({1, 0}) << ' ' << C.at({1, 1}) << '\n';
    return 0;
}