#ifndef ARNOLDI_HPP
#define ARNOLDI_HPP
#include<vector>
#include <mkl.h>
void ARNOLDI(sparse_matrix_t A,
             const std::vector<double>& r, 
             int& m, 
             std::vector<double>& V, 
             std::vector<double>& H);

#endif