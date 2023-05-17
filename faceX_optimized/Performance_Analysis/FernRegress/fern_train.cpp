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

#include "Covariance.h" //for covariance ispc


#include "fern_train.h"

#include<iostream>
#include<cstdlib>
#include <stdlib.h> 
#include<memory>
#include<algorithm>
#include<string.h>
#include<mpi.h>
#include <omp.h>
#include <x86intrin.h>
#include "logtime_for_MPI.h"

using namespace std;

FernTrain::FernTrain(const TrainingParameters &tp) : training_parameters(tp)
{
}


void FernTrain::Regress(const vector<vector<cv::Point2d>> *targets, 
	cv::Mat pixels_val, cv::Mat pixels_cov)
{
	int process_rank, process_cnt;
	MPI_Comm_size(MPI_COMM_WORLD, &process_cnt);
	MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
	const bool isroot = (0 == process_rank);

MPI_Barrier(MPI_COMM_WORLD);
const double t1 = MPI_Wtime();
	const int f = training_parameters.F / process_cnt;
	const int remainder = training_parameters.F % process_cnt;
	int f_offset[process_cnt];
	int f_workload[process_cnt];
	for (int i=0;i<remainder; i++){
		f_workload[i] = 3*f+3;
		f_offset[i] = 3*(i*f+i);
	}
	for (int i=remainder;i<process_cnt; i++){
		f_workload[i] = 3*f;
		f_offset[i] = 3*(i*f+remainder);
	}

	cv::Mat Y(targets->size(), (*targets)[0].size() * 2, CV_64FC1);
	for (int i = 0; i < Y.rows; ++i)
	{
		for (int j = 0; j < Y.cols; j += 2)
		{
			Y.at<double>(i, j) = (*targets)[i][j / 2].x;
			Y.at<double>(i, j + 1) = (*targets)[i][j / 2].y;
		}
	}
	features_index.assign(training_parameters.F, pair<int, int>());
	thresholds.assign(training_parameters.F, 0);
	
	double buf[training_parameters.F * 3];
	double sbuf[training_parameters.F * 3];
	for (int i = 0; i < f_workload[process_rank]; i+=3)
	{
		cv::Mat projection(Y.cols, 1, CV_64FC1);
		cv::theRNG().fill(projection, cv::RNG::NORMAL, cv::Scalar(0), cv::Scalar(1));
		unique_ptr<double[]> Y_proj_data(new double[Y.rows + 3]);
		cv::Mat Y_proj(Y.rows, 1, CV_64FC1, cv::alignPtr(Y_proj_data.get(), 32));
		static_cast<cv::Mat>(Y * projection).copyTo(Y_proj);
		/*double Y_proj_cov = CovarianceZY0(Y_proj.ptr<double>(0), 
			Y_proj.ptr<double>(0), Y_proj.total());	// Use AVXCovariance if you want.*/
		double Y_proj_cov = CovarianceISPC(Y_proj.ptr<double>(0), 
			Y_proj.ptr<double>(0), Y_proj.total());
		/*double Y_proj_cov = CovarianceZYIntel(Y_proj.ptr<double>(0), 
			Y_proj.ptr<double>(0), Y_proj.total());	*/
		vector<double> Y_pixels_cov(pixels_val.rows);
		for (int j = 0; j < pixels_val.rows; ++j)
		{
			/*Y_pixels_cov[j] = CovarianceZY0(Y_proj.ptr<double>(0), 
				pixels_val.ptr<double>(j), Y_proj.total());	// Use AVXCovariance if you want.*/
			Y_pixels_cov[j] = CovarianceISPC(Y_proj.ptr<double>(0), 
				pixels_val.ptr<double>(j), Y_proj.total());
			/*Y_pixels_cov[j] = CovarianceZYIntel(Y_proj.ptr<double>(0), 
				pixels_val.ptr<double>(j), Y_proj.total());*/
		}
		double max_corr = -1;
        int ind1=0, ind2=0;
#pragma omp parallel
{
		double t_max_corr = -1;
		int idx1=0, idx2=0;
#pragma omp for schedule(static, 8) collapse(2) nowait
		for (int j = 0; j < pixels_val.rows; ++j)
		{
			for (int k = 0; k < pixels_val.rows; ++k)
			{
				double corr = (Y_pixels_cov[j] - Y_pixels_cov[k]) / sqrt(
					Y_proj_cov * (pixels_cov.at<double>(j, j) 
					+ pixels_cov.at<double>(k, k) 
					- 2 * pixels_cov.at<double>(j, k)));
				if (corr > t_max_corr)
				{
					t_max_corr = corr;
					idx1 = j;
					idx2 = k;
				}
			}
		}
#pragma omp critical
{
		if (t_max_corr > max_corr)
		{
			max_corr = t_max_corr;
			ind1 = idx1;
			ind2 = idx2;
		}
}		
}
		
		double threshold_max = -1000000;
		double threshold_min = 1000000;
        
#pragma omp parallel 
{
		__m256d t_min = _mm256_set1_pd(threshold_min), t_max = _mm256_set1_pd(threshold_max);
#pragma omp for schedule(static) nowait
		for (int j = 0; j < pixels_val.cols; j+=4)
		{
			__m256d p1 = _mm256_load_pd(&pixels_val.at<double>(ind1, j));
			__m256d val = _mm256_sub_pd(p1, _mm256_load_pd(&pixels_val.at<double>(ind2, j)));
			t_max = _mm256_max_pd(t_max, val);
			t_min = _mm256_min_pd(t_min, val);
		}
		double *pmax = reinterpret_cast<double*>(&t_max),*pmin = reinterpret_cast<double*>(&t_min);
		double lmax =  max({pmax[0],pmax[1], pmax[2],pmax[3]});
		double lmin =  min({pmin[0],pmin[1], pmin[2],pmin[3]});
#pragma omp atomic update, compare
		if(lmax > threshold_max) {threshold_max = lmax;} 
#pragma omp atomic update, compare
		if(lmin < threshold_min) {threshold_min = lmin;} 
}
        sbuf[i] = ind1;
        sbuf[i+1] = ind2;
		sbuf[i+2] = ((threshold_max + threshold_min) / 2
			+ cv::theRNG().uniform(-(threshold_max - threshold_min) * 0.1, 
			(threshold_max - threshold_min) * 0.1));
	}

	MPI_Allgatherv(&sbuf[0], f_workload[process_rank],
		MPI_DOUBLE, &buf[0], f_workload, f_offset, MPI_DOUBLE,MPI_COMM_WORLD);

	for(int i=0; i<training_parameters.F; i++){
		features_index[i].first = (int)round(buf[3*i]);
		features_index[i].second = (int)round(buf[i*3+1]);
		thresholds[i] = buf[i*3+2];
	}

	int outputs_count = 1 << training_parameters.F;
	outputs.assign(outputs_count, vector<cv::Point2d>((*targets)[0].size()));
	vector<int> each_output_count(outputs_count);

	int s = process_rank * (targets->size() / process_cnt);
	int t = min(targets->size(), s+targets->size() / process_cnt);
	const int block_size = 11;
	const int buf_size = outputs_count * block_size;
	double* sendbuf = (double*) calloc(buf_size, sizeof(double));
	double* recvbuf = (double*) calloc(buf_size, sizeof(double));
	

#pragma omp parallel for schedule(static) reduction(+: sendbuf[:buf_size])
	for (int i = s; i < t; ++i)
	{
		int mask = 0;
		for (int j = 0; j < training_parameters.F; ++j)
		{
			double p1 = pixels_val.at<double>(features_index[j].first, i);
			double p2 = pixels_val.at<double>(features_index[j].second, i);
			mask |= (p1 - p2 > thresholds[j]) << j;
		}
		for (int j = 0; j < 5; ++j)
		{
			sendbuf[mask * block_size + 2*j] += (*targets)[i][j].x;
			sendbuf[mask * block_size + 2*j+1] += (*targets)[i][j].y;
		}
		sendbuf[mask * block_size + block_size - 1] += 1;
	}
	MPI_Allreduce(&sendbuf[0], &recvbuf[0], buf_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	for (int i = 0; i < outputs_count; ++i)
	{
		double factor = recvbuf[i*block_size+block_size-1];
		factor = 1.0 / (factor + training_parameters.Beta);
		for (int j=0; j<5;j++){
			outputs[i][j].x = recvbuf[i*block_size+2*j] * factor;
			outputs[i][j].y = recvbuf[i*block_size+2*j + 1] * factor;
		}
	}
const double t2 = MPI_Wtime();
logtime_for_MPI(t2-t1,"Fern_train"); 
free(sendbuf);
free(recvbuf);
sendbuf=NULL;
recvbuf=NULL;
}

vector<cv::Point2d> FernTrain::Apply(cv::Mat features)const
{
	int outputs_index = 0;
	for (int i = 0; i < training_parameters.F; ++i)
	{
		pair<int, int> feature = features_index[i];
		double p1 = features.at<double>(feature.first);
		double p2 = features.at<double>(feature.second);
		outputs_index |= (p1 - p2 > thresholds[i]) << i;
	}
	return outputs[outputs_index];
}

void FernTrain::ApplyMini(cv::Mat features, std::vector<double> &coeffs)const
{
	int outputs_index = 0;
	for (int i = 0; i < training_parameters.F; ++i)
	{
		pair<int, int> feature = features_index[i];
		double p1 = features.at<double>(feature.first);
		double p2 = features.at<double>(feature.second);
		outputs_index |= (p1 - p2 > thresholds[i]) << i;
	}

	const vector<pair<int, double>> &output = outputs_mini[outputs_index];
	for (int i = 0; i < training_parameters.Q; ++i)
		coeffs[output[i].first] += output[i].second;
}



void FernTrain::write(cv::FileStorage &fs)const
{
	fs << "{";
	fs << "thresholds" << thresholds;
	fs << "features_index";
	fs << "[";
	for (auto it = features_index.begin(); it != features_index.end(); ++it)
		fs << "{" << "first" << it->first << "second" << it->second << "}";
	fs << "]";
	fs << "outputs_mini";
	fs << "[";
	for (const auto &output: outputs_mini)
	{
		fs << "[";
		for (int i = 0; i < training_parameters.Q; ++i)
		{
			fs << "{" << "index" << output[i].first <<
				"coeff" << output[i].second << "}";
		}
		fs << "]";
	}
	fs << "]";
	fs << "}";
}

void write(cv::FileStorage& fs, const string&, const FernTrain &f)
{
	f.write(fs);
}
