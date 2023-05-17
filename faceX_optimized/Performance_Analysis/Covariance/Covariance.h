#pragma once
#include <immintrin.h>


extern "C" double CovarianceISPC(const double *x, const double * y, const int size);
double CovarianceAVX2(const double * __restrict__ x, const double * __restrict__ y, const int size);
double Covariance(double *x, double * y, const int size);

