#include <iostream>
#include "tensor.h"
#include "matmul.h"

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
         Tensor C = matmul(A, B);
         std::cout << C.at({1,1}) << std::endl;
    return 0;
}