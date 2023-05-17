/*
FaceX-Train is a tool to train model file for FaceX, which is an open
source face alignment library.

Copyright(C) 2015  Yang Cao

This program is free software : you can redistribute it and / or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.If not, see <http://www.gnu.org/licenses/>.
*/

#include "utils_train.h"
#include <immintrin.h>
#include<cmath>
#include<iostream>
#include<cassert>
#include <Eigen/Dense>
#include <mpi.h>
#include <cblas.h>
#include <omp.h>

using namespace std;

void Transform::Apply(vector<cv::Point2d> *x, bool need_translation)
{
	for (cv::Point2d &p : *x)
	{
		cv::Matx21d v;
		v(0) = p.x;
		v(1) = p.y;
		v = scale_rotation * v;
		if (need_translation)
			v += translation;
		p.x = v(0);
		p.y = v(1);
	}
}

Transform Procrustes(const vector<cv::Point2d> &x, const vector<cv::Point2d> &y)
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

	Transform result;
	result.scale_rotation(0, 0) = solution(0);
	result.scale_rotation(0, 1) = -solution(1);
	result.scale_rotation(1, 0) = solution(1);
	result.scale_rotation(1, 1) = solution(0);
	result.translation(0) = solution(2);
	result.translation(1) = solution(3);
	return result;
}

/// ---------------------------- Procrustes SIMD Implementation ------------------------------------- ///
double _mm256_get1_pd(__m256d vec, int idx);

Transform ProcrustesV(const std::vector<cv::Point2d> &x, const std::vector<cv::Point2d> &y)
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

    Transform result;
    result.scale_rotation(0, 0) = solution(0);
    result.scale_rotation(0, 1) = -solution(1);
    result.scale_rotation(1, 0) = solution(1);
    result.scale_rotation(1, 1) = solution(0);
    result.translation(0) = solution(2);
    result.translation(1) = solution(3);
    return result;
}

//helper function that gets a double value from the idx-th position of a 256-bit vector
double _mm256_get1_pd(__m256d vec, int idx) {
    double res[4];
    _mm256_storeu_pd(res, vec);
    return res[idx];
}
/// ---------------------------- Implementation End ------------------------------------- ///

static void Normalize(vector<cv::Point2d> * shape, const TrainingParameters &tp)
{

	cv::Point2d center;
	for (const cv::Point2d p : *shape)
		center += p;
	center *= 1.0 / shape->size();
	for (cv::Point2d &p : *shape)
		p -= center;

	cv::Point2d left_eye = (*shape).at(tp.left_eye_index);
	cv::Point2d right_eye = (*shape).at(tp.right_eye_index);
	double eyes_distance = cv::norm(left_eye - right_eye);
	double scale = 1 / eyes_distance;

	double theta = -atan((right_eye.y - left_eye.y) / (right_eye.x - left_eye.x));

	// Must do translation first, and then rotation.
	// Therefore, translation is done separately
	Transform t;
	t.scale_rotation(0, 0) = scale * cos(theta);
	t.scale_rotation(0, 1) = -scale * sin(theta);
	t.scale_rotation(1, 0) = scale * sin(theta);
	t.scale_rotation(1, 1) = scale * cos(theta);


	t.Apply(shape, false);
}


vector<cv::Point2d> MeanShape(vector<vector<cv::Point2d>> shapes, 
	const TrainingParameters &tp)
{
	const int kIterationCount = 10;
	vector<cv::Point2d> mean_shape = shapes[0];
	
	for (int i = 0; i < kIterationCount; ++i)
	{
		for (vector<cv::Point2d> & shape: shapes)
		{
			Transform t = ProcrustesV(mean_shape, shape);
			t.Apply(&shape);
		}
		
		for (cv::Point2d & p : mean_shape)
			p.x = p.y = 0;
		
		for (const vector<cv::Point2d> & shape : shapes)
			for (int j = 0; j < mean_shape.size(); ++j)
			{
				mean_shape[j].x += shape[j].x;
				mean_shape[j].y += shape[j].y;
			}

		for (cv::Point2d & p : mean_shape)
			p *= 1.0 / shapes.size();

		Normalize(&mean_shape, tp);
	}

	return mean_shape;
}


vector<cv::Point2d> ShapeDifference(const vector<cv::Point2d> &s1,
	const vector<cv::Point2d> &s2)
{
	assert(s1.size() == s2.size());
	vector<cv::Point2d> result(s1.size());
	for (int i = 0; i < s1.size(); ++i)
		result[i] = s1[i] - s2[i];
	return result;
}

vector<cv::Point2d> ShapeAdjustment(const vector<cv::Point2d> &shape,
	const vector<cv::Point2d> &offset)
{
	assert(shape.size() == offset.size());
	vector<cv::Point2d> result(shape.size());
	for (int i = 0; i < shape.size(); ++i)
		result[i] = shape[i] + offset[i];
	return result;
}

double Covariance(double *x, double * y, const int size)
{
	double a = 0, b = 0, c = 0;
	for (int i = 0; i < size; ++i)
	{
		a += x[i];
		b += y[i];
		c += x[i] * y[i];
	}

	return c / size - (a / size) * (b / size);
}

/*
The function below has the same functionality as the one above. It uses 
AVX2 instructions to speed up computation. Use it if your processor is 
newer than or equal to Intel Haswell, and your compiler support AVX2.
*/

/*
#include<intrin.h>

double Covariance(double *x, double * y, const int size)
{
	const int kBlockSize = 4;
	const int kBlocksCount = size / kBlockSize;

	double *p = x, *q = y;
	__m256d sum1 = _mm256_setzero_pd();
	__m256d sum2 = _mm256_setzero_pd();
	__m256d sum = _mm256_setzero_pd();
	__m256d load1, load2;
	for (int i = 0; i < kBlocksCount ; ++i, p += kBlockSize, q += kBlockSize)
	{
		load1 = _mm256_load_pd(p);
		load2 = _mm256_load_pd(q);
		sum1 = _mm256_add_pd(sum1, load1);
		sum2 = _mm256_add_pd(sum2, load2);
		sum = _mm256_fmadd_pd(load1, load2, sum);
	}
	p = reinterpret_cast<double*>(&sum1);
	double mean_x = (p[0] + p[1] + p[2] + p[3]) / size;

	q = reinterpret_cast<double*>(&sum2);
	double mean_y = (q[0] + q[1] + q[2] + q[3]) / size;

	p = reinterpret_cast<double*>(&sum);
	double result = (p[0] + p[1] + p[2] + p[3]) / size;
	return result - mean_x * mean_y;
}*/

vector<cv::Point2d> MapShape(cv::Rect original_face_rect,
	const vector<cv::Point2d> original_landmarks, cv::Rect new_face_rect)
{
	vector<cv::Point2d> result;
	for (const cv::Point2d &landmark: original_landmarks)
	{
		result.push_back(landmark);
		result.back() -= cv::Point2d(original_face_rect.x, original_face_rect.y);
		result.back().x *= 
			static_cast<double>(new_face_rect.width) / original_face_rect.width;
		result.back().y *= 
			static_cast<double>(new_face_rect.height) / original_face_rect.height;
		result.back() += cv::Point2d(new_face_rect.x, new_face_rect.y);
	}
	return result;
}

//// ----------------------------------- Eigen Implementation ------------------------------------- ///

std::vector<std::pair<int, double>> OMP_Eigen(cv::Mat x, cv::Mat base, int coeff_count)
{

	vector<pair<int, double>> result;
    Eigen::MatrixXd X = Eigen::MatrixXd::Map(x.ptr<double>(), x.rows, x.cols);
    Eigen::MatrixXd residual = Eigen::MatrixXd::Map(x.ptr<double>(), x.rows, x.cols);
    Eigen::MatrixXd baseEigen = Eigen::MatrixXd::Map(base.ptr<double>(), base.rows, base.cols);

    for (int i = 0; i < coeff_count; ++i)
    {
        int max_index = 0;
        double max_value = 0;

        for (int j = 0; j < base.cols; ++j)
        {
            Eigen::MatrixXd column = baseEigen.col(j);
			double current_value = (residual.transpose() * column).cwiseAbs().sum();

            if (current_value > max_value)
            {
                max_value = current_value;
                max_index = j;
            }
        }

        result.push_back(make_pair(max_index, 0));
        Eigen::MatrixXd sparse_base(base.rows, result.size());

        for (int j = 0; j < result.size(); ++j)
    		sparse_base.col(j) = baseEigen.col(result[j].first);

        Eigen::MatrixXd lhs = sparse_base.transpose() * sparse_base;
        Eigen::MatrixXd rhs = sparse_base.transpose() * X;
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(lhs, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::MatrixXd beta = svd.solve(rhs);

        for (int j = 0; j < result.size(); ++j)
            result[j].second = beta(j);

        residual -= sparse_base * beta;
    }

        for (const auto& pair : result) {
        int key = pair.first;
        double value = pair.second;
        //std::cout << "Key: " << key << ", Value: " << value << "eigen - library"<< std::endl;
    }

    return result;
}

//// ---------------------------------------------------------------------------------------------------- ///


vector<pair<int, double>> OMP(cv::Mat x, cv::Mat base, int coeff_count)
{
	vector<pair<int, double>> result;
	cv::Mat residual = x.clone();
	for (int i = 0; i < coeff_count; ++i)
	{
		int max_index = 0;
		double max_value = 0;
		for (int j = 0; j < base.cols; ++j)
		{
			double current_value = abs(
				static_cast<cv::Mat>(residual.t() * base.col(j)).at<double>(0));
			if (current_value > max_value)
			{
				max_value = current_value;
				max_index = j;
			}
		}

		result.push_back(make_pair(max_index, 0));
		cv::Mat sparse_base(base.rows, result.size(), CV_64FC1);
		for (int j = 0; j < result.size(); ++j)
			base.col(result[j].first).copyTo(sparse_base.col(j));


		// The cv::solve function solves the linear system LHS * beta = RHS and stores the result in the beta matrix. 
		cv::Mat beta;
		cv::solve(sparse_base.t() * sparse_base, 
			sparse_base.t() * x, beta, cv::DECOMP_SVD);
		for (int j = 0; j < result.size(); ++j)
			result[j].second = beta.at<double>(j);
		residual -= sparse_base * beta;
	}
        for (const auto& pair : result) {
        int key = pair.first;
        double value = pair.second;
        //std::cout << "Key: " << key << ", Value: " << value << "OMP - library" << std::endl;
    }


	return result;
}


//// -------------------------------------- MPI Implementation ------------------------------------- ///


vector<pair<int, double>> OMP_sub_routine(cv::Mat x, cv::Mat base, int coeff_count)
{
	vector<pair<int, double>> result;
	cv::Mat residual = x.clone();

	for (int i = 0; i < coeff_count; ++i)
	{
		int max_index = 0;
		double max_value = 0;

		for (int j = 0; j < base.cols; ++j)
		{
			double current_value = abs(
				static_cast<cv::Mat>(residual.t() * base.col(j)).at<double>(0));
			if (current_value > max_value)
			{
				max_value = current_value;
				max_index = j;
			}
		}

		result.push_back(make_pair(max_index, 0));
		cv::Mat sparse_base(base.rows, result.size(), CV_64FC1);
		for (int j = 0; j < result.size(); ++j)
			base.col(result[j].first).copyTo(sparse_base.col(j));

		cv::Mat beta;
		cv::solve(sparse_base.t() * sparse_base, 
			sparse_base.t() * x, beta, cv::DECOMP_SVD);

		for (int j = 0; j < result.size(); ++j)
			result[j].second = beta.at<double>(j);
		residual -= sparse_base * beta;
	}
	return result;
}

vector<pair<int, double>> OMP_MPI(cv::Mat x, cv::Mat base, int coeff_count)
{
    int numProcesses, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	vector<pair<int, double>> result;

    int start = rank * (coeff_count / numProcesses);

    // To summarize, the line calculates the value of end based on the rank of the process.
    // If the process is the last one, end is set to coeff_count. Otherwise, it is set to (rank + 1) * (coeff_count / numProcesses),
    int end = (rank == numProcesses - 1) ? coeff_count : (rank + 1) * (coeff_count / numProcesses);
    vector<pair<int, double>> local_results = OMP_sub_routine(x, base, end - start);

    // This is the number of objects stored in the object returned by the OMP_sub_routine in vector<pair<int, double>>
    int local_size = local_results.size();
    // printf("rank =%d local_size =%d \n",rank,local_size);
    vector<double> local_buffer;
	vector<double> buffer;
	buffer.reserve(coeff_count * 2);

	// Serialize data
	local_buffer.reserve(local_size * 2);  // Reserve space for pair<int, double> elements

	for (const auto& pair : local_results) {
		local_buffer.push_back(static_cast<double>(pair.first));
		local_buffer.push_back(pair.second);
        }
    
	int local_buffer_size = local_size * 2;

    // Gather the serialized buffer onto rank 0
    vector<int> buffer_sizes(numProcesses);
    vector<int> buffer_displs(numProcesses);

	//std::cout << 0 << std::endl;


	MPI_Gather(&local_buffer_size, 1, MPI_INT, buffer_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

	//std::cout << 0 << std::endl;

	// calculate displacement
	int max_size = 0;
	buffer_displs[0] =0;
	for (int i = 1; i < numProcesses; ++i) {
		buffer_displs[i] = max_size;
		max_size += buffer_sizes[i];
		}
    // printf("rank =%d, max_size =%d\n", rank,max_size);
    MPI_Gatherv(local_buffer.data(), local_buffer_size, MPI_DOUBLE, buffer.data(),
                buffer_sizes.data(), buffer_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//std::cout << 1 << std::endl;

	for (int i = 0; i < buffer.size(); i += 2) {
		int first = static_cast<int>(buffer[i]);
		double second = buffer[i + 1];
		result.push_back(make_pair(first, second));
	}

	//std::cout << 2 << std::endl;

	return result;
}
//// ------------------------------------------------------------------------------------------- ///


string TrimStr(const string &s, const string &space)
{
	string result = s;
	result = result.erase(0, result.find_first_not_of(space));
	return result.erase(result.find_last_not_of(space) + 1);
}
