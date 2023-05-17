
#include "Covariance.h"


// This function calculates the covariance between two arrays using the AVX2 instruction set to perform vectorized operations.
double CovarianceAVX2(const double * __restrict__ x, const double * __restrict__ y, const int size)
{
    // Define the size of the vectors that will be used for vectorized operations.
    const size_t simd_width = 32 / sizeof(double);
    __m256d sum_a = _mm256_setzero_pd();
    __m256d sum_b = _mm256_setzero_pd();
    __m256d sum_c = _mm256_setzero_pd();

    // Iterate over the elements of the arrays, processing them in chunks of size 'simd_width'.
    for (size_t i = 0; i < size; i += simd_width) 
    {
        // Load the next chunk of 'x' and 'y' into AVX2 registers.
        __m256d x_vec = _mm256_load_pd(x + i);
        __m256d y_vec = _mm256_load_pd(y + i);
        
        // Update the values of the three vectors using AVX2 instructions.
        sum_a = _mm256_add_pd(sum_a, x_vec);
        sum_b = _mm256_add_pd(sum_b, y_vec);
        sum_c = _mm256_fmadd_pd(x_vec, y_vec, sum_c);
    }

    // Store the values of the three vectors into arrays of size 'simd_width'.
    double a_array[simd_width], b_array[simd_width], c_array[simd_width];
    _mm256_storeu_pd(a_array, sum_a);
    _mm256_storeu_pd(b_array, sum_b);
    _mm256_storeu_pd(c_array, sum_c);

    // Calculate the final values of 'a', 'b', and 'c'
    double a = a_array[0] + a_array[1] + a_array[2] + a_array[3];
    double b = b_array[0] + b_array[1] + b_array[2] + b_array[3];
    double c = c_array[0] + c_array[1] + c_array[2] + c_array[3];

    // Process the remaining elements if the size is not divisible by 4
    for (int i = size - size % simd_width; i < size; i++) 
    {
        a += x[i];
        b += y[i];
        c += x[i] * y[i];
    }

    return c / size - (a / size) * (b / size);
}
