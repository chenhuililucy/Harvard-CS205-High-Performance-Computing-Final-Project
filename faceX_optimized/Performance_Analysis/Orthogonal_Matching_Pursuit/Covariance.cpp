
#include "Covariance.h"
#include "mpi_global_vars.h"

#include <iostream>
#include <fstream>
#include <cstdlib>


#include <cassert>

#include <mpi.h>
#include <omp.h>


double CovarianceZY0(double *x, double * y, const int size)
{
	double a = 0, b = 0, c = 0;
	for (int i = 0; i < size; ++i)
	{
		a += x[i];
		b += y[i];
		c += x[i] * y[i];
	}
	
	double ans0 = c / size - (a / size) * (b / size);
	
	return ans0;
}

double * CovarianceZY00(const double *x, const double * y, const int size)
{
	double a = 0, b = 0, c = 0;
	for (int i = 0; i < size; ++i)
	{
		a += x[i];
		b += y[i];
		c += x[i] * y[i];
	}
	
	double* ans = new double[3];
	ans[0] = a;
	ans[1] = b;
	ans[2] = c;
	return ans;
}

double CovarianceZY1(const double *x, const double * y, const int size){
	//MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
	//MPI_Comm_size(MPI_COMM_WORLD, &process_cnt);
	// Compute local partial sums of a, b, and c for this process's chunk of the data
	double a = 0., b = 0., c = 0.;
	
	const int chunk_size = size / process_cnt;
	const int start_idx = process_rank * chunk_size;
	const int end_idx = (process_rank == process_cnt - 1) ? size : start_idx + chunk_size;
	const int rank_ele_cnt = end_idx - start_idx + 1;
	
	double *arr = CovarianceZY00(&x[start_idx], &y[start_idx], rank_ele_cnt);
	a = arr[0];
	b = arr[1];
	c = arr[2];

	double global_a, global_b, global_c;
	MPI_Reduce(&a, &global_a, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&b, &global_b, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&c, &global_c, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (process_rank == 0) {
		// Compute the final covariance value using the global partial sums
		double ans = global_c / size - (global_a / size) * (global_b / size);
		return ans;
	} else {
		return 0;
	}
	
}