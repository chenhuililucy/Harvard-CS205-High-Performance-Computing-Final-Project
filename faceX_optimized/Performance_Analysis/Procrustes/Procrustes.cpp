//Filename:  Procrustes.cpp
#include <immintrin.h>
#include<cmath>
#include<iostream>
#include<cassert>
#include <omp.h>
#include <chrono>
#include <vector>
#include <string>
#include <utility>
#include <opencv2/opencv.hpp>

using namespace std;

template<typename T>
inline T Sqr(T a)
{
	return a * a;
}

cv::Matx41d Procrustes(const vector<cv::Point2d> &x, const vector<cv::Point2d> &y)
{
	assert(x.size() == y.size());
	int landmark_count = x.size();
	double X1 = 0, X2 = 0, Y1 = 0, Y2 = 0, Z = 0, W = landmark_count;
	double C1 = 0, C2 = 0;

	for (int i = 0; i < landmark_count; ++i)
	{
		X1 += x[i].x;
		X2 += y[i].x;
		Y1 += x[i].y;
		Y2 += y[i].y;
		Z += Sqr(y[i].x) + Sqr(y[i].y);
		C1 += x[i].x * y[i].x + x[i].y * y[i].y;
		C2 += x[i].y * y[i].x - x[i].x * y[i].y;
	}

	cv::Matx44d A(X2, -Y2, W, 0,
		Y2, X2, 0, W,
		Z, 0, X2, Y2,
		0, Z, -Y2, X2);
	cv::Matx41d b(X1, Y1, C1, C2);
	cv::Matx41d solution = A.inv() * b;
    return solution;
	
}

/// ---------------------------- Procrustes SIMD Implementation ------------------------------------- ///
double _mm256_get1_pd(__m256d vec, int idx);

cv::Matx41d ProcrustesV(const std::vector<cv::Point2d> &x, const std::vector<cv::Point2d> &y)
{
    assert(x.size() == y.size());
    int landmark_count = x.size();
    double X1 = 0, X2 = 0, Y1 = 0, Y2 = 0, Z = 0, W = landmark_count;
    double C1 = 0, C2 = 0;

    __m256d sumX1 = _mm256_setzero_pd();
    __m256d sumX2 = _mm256_setzero_pd();
    __m256d sumY1 = _mm256_setzero_pd();
    __m256d sumY2 = _mm256_setzero_pd();
    __m256d sumZ = _mm256_setzero_pd();
    __m256d sumC1 = _mm256_setzero_pd();
    __m256d sumC2 = _mm256_setzero_pd();

    for (int i = 0; i < landmark_count/2; i += 2)
    {
        __m128d x1 = _mm_loadu_pd(&x[i*2].x);
        __m128d y1 = _mm_loadu_pd(&x[i*2].y);
        __m128d x2 = _mm_loadu_pd(&y[i*2].x);
        __m128d y2 = _mm_loadu_pd(&y[i*2].y);

        sumX1 = _mm256_add_pd(sumX1, _mm256_castpd128_pd256(x1));
        sumY1 = _mm256_add_pd(sumY1, _mm256_castpd128_pd256(y1));
        sumX2 = _mm256_add_pd(sumX2, _mm256_castpd128_pd256(x2));
        sumY2 = _mm256_add_pd(sumY2, _mm256_castpd128_pd256(y2));

        __m256d sqrX2 = _mm256_mul_pd(_mm256_castpd128_pd256(x2), _mm256_castpd128_pd256(x2));
        __m256d sqrY2 = _mm256_mul_pd(_mm256_castpd128_pd256(y2), _mm256_castpd128_pd256(y2));

        sumZ = _mm256_add_pd(sumZ, _mm256_add_pd(sqrX2, sqrY2));

        __m256d prod1 = _mm256_mul_pd(_mm256_castpd128_pd256(x1), _mm256_castpd128_pd256(x2));
        __m256d prod2 = _mm256_mul_pd(_mm256_castpd128_pd256(y1), _mm256_castpd128_pd256(y2));

        sumC1 = _mm256_add_pd(sumC1, _mm256_add_pd(prod1, prod2));

        __m256d prod3 = _mm256_mul_pd(_mm256_castpd128_pd256(y1), _mm256_castpd128_pd256(x2));
        __m256d prod4 = _mm256_mul_pd(_mm256_castpd128_pd256(x1), _mm256_castpd128_pd256(y2));

        sumC2 = _mm256_add_pd(sumC2, _mm256_sub_pd(prod3, prod4));
    }

    X1 += _mm256_get1_pd(sumX1, 0) + _mm256_get1_pd(sumX1, 2);
    Y1 += _mm256_get1_pd(sumY1, 0) + _mm256_get1_pd(sumY1, 2);
    X2 += _mm256_get1_pd(sumX2, 0) + _mm256_get1_pd(sumX2, 2);
    Y2 += _mm256_get1_pd(sumY2, 0) + _mm256_get1_pd(sumY2, 2);
    Z += _mm256_get1_pd(sumZ, 0) + _mm256_get1_pd(sumZ, 2);
    C1 += _mm256_get1_pd(sumC1, 0) + _mm256_get1_pd(sumC1, 2);
    C2 += _mm256_get1_pd(sumC2, 0) + _mm256_get1_pd(sumC2, 2);

    cv::Matx44d A(X2, -Y2, W, 0,
                  Y2, X2, 0, W,
                  Z, 0, X2, Y2,
                  0, Z, -Y2, X2);
    cv::Matx41d b(X1, Y1, C1, C2);
    cv::Matx41d solution = A.inv() * b;
    return solution;
}

//helper function that gets a double value from the idx-th position of a 256-bit vector
double _mm256_get1_pd(__m256d vec, int idx) {
    double res[4];
    _mm256_storeu_pd(res, vec);
    return res[idx];
}
/// ---------------------------- Implementation End ------------------------------------- ///


// Generates random input data for the Procrustes functions.
void generate_input_data(std::vector<cv::Point2d>& x, std::vector<cv::Point2d>& y, int size) {
    x.resize(size);
    y.resize(size);
    for (int i = 0; i < size; ++i) {
        x[i] = cv::Point2d(rand(), rand());
        y[i] = cv::Point2d(rand(), rand());
    }
}

// Measures the execution time of a given function with a given set of input data.
template <typename Func>
double measure_execution_time(Func&&func, const std::vector<cv::Point2d>& x, const std::vector<cv::Point2d>& y) {
    auto start_time = std::chrono::steady_clock::now();
    func(x, y);
    auto end_time = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
}

//The size of the input data in this test is data_size.
//The generate_input_data function creates two vectors of 2D points, x and y, 
//with data_size elements each. Therefore, each call to the Procrustes and 
//ProcrustesV functions in the test_procrustes_performance function will receive 
//two vectors of 2D points as input.
// Tests the Procrustes and ProcrustesV functions with random input data and compare the results.
// Measures the runtime of both functions.
void test_procrustes_performance(int data_size) {
    const int num_tests = 10;
    double total_time_procrustes = 0;
    double total_time_procrustes_v = 0;

    cv::Matx41d solution_procrustes;
    cv::Matx41d solution_procrustes_v;
    for (int i = 0; i < num_tests; ++i) {
        std::vector<cv::Point2d> x, y;
        generate_input_data(x, y, data_size);
        
        // Compute the solutions using Procrustes and ProcrustesV
        cv::Matx41d solution_procrustes = Procrustes(x, y);
        cv::Matx41d solution_procrustes_v = ProcrustesV(x, y);
        total_time_procrustes += measure_execution_time(Procrustes, x, y);
        total_time_procrustes_v += measure_execution_time(ProcrustesV, x, y);
    }
    double avg_time_procrustes = total_time_procrustes / num_tests;
    double avg_time_procrustes_v = total_time_procrustes_v / num_tests;


    std::cout << "Data Size: " << data_size << " 2dPoint\n";
    std::cout << "Average execution time of Procrustes: " << avg_time_procrustes << " ms\n";
    std::cout << "Average execution time of ProcrustesV: " << avg_time_procrustes_v << " ms\n";
    // Compare the solutions
    if (solution_procrustes == solution_procrustes_v) {
        std::cout << "Procrustes and ProcrustesV produce the same result\n";
    } else {
        std::cout << "Procrustes and ProcrustesV produce different results\n";
    }
}


int main()
{
    // Test Procrustes and ProcrustesV performance
    std::vector<int> data_sizes = {10000000, 20000000, 30000000, 40000000};
    for (int data_size : data_sizes) {
        test_procrustes_performance(data_size);
        std::cout << std::endl;
    }
    std::cout << "[roofline analysis](https://code.harvard.edu/CS205/team06/blob/main/faceX_optimized/Performance_Analysis/Procrustes/procrustes_roofline.png)\n";
    std::cout << "[scaling analysis](https://code.harvard.edu/CS205/team06/blob/main/faceX_optimized/Performance_Analysis/Procrustes/procrustes_scaling.png)\n";
    return 0;

}

/*
FLOPs of Procrustes:
Inside the for loop, which iterates landmark_count times, we have the following operations:

a. 4 addition operations: X1, X2, Y1, Y2
b. 2 multiplication and addition operations: Z
c. 4 multiplication and 2 addition operations: C1
d. 4 multiplication and 1 subtraction operation: C2

Total FLOPs for each loop iteration = 4 (additions) + 2 (multiplications and additions) 
                                    + 4 (multiplications) + 2 (additions) 
                                    + 4 (multiplications) + 1 (subtraction) 
                                    = 17

Total FLOPs = 17 * landmark_count
*/

/*
FLOPs of ProcrustesV:

Inside the for loop, we have multiple SIMD operations that need to be considered,
where each SIMD operation can process 2 double-precision numbers in parallel.

Note that the loop iterates landmark_count/2 times.

a. 6 addition operations: sumX1, sumY1, sumX2, sumY2, sumC1, sumC2
b. 8 multiplication operations: sqrX2, sqrY2, prod1, prod2, prod3, prod4
c. 1 subtraction operation: sumC2

After the loop, there are 6 addition operations to sum the values.

Total FLOPs = (6 additions + 8 multiplications + 1 subtraction) * 2 (SIMD operations) * (landmark_count/2) (loop iterations) 
            + 6 (additions after the loop)
            = (15 * 2) * (landmark_count/2) + 6
            = 30 * (landmark_count/2) + 6
Total FLOPs = 15 * landmark_count + 6
*/