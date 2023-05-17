
/*The RegressorTrain class initializes with TrainingParameters and a vector of FernTrain objects. 
The regressor is trained by calling the RegressorTrain::Regress() function which takes in the mean shape,
the training targets, and the training data. The function iterates over the pixels_ vector to create a matrix of pixel values pixels_val and compute its covariance matrix pixels_cov.
The regressor then trains each fern using the FernTrain::Regress() function and applies it to the pixels_val matrix. 
After that, the regressor applies the shape difference to each target vector and compresses the ferns using the CompressFerns() function.*/

#include "regressor_train.h"

#include <utility>
#include <iostream>
#include <memory>
#include <algorithm>
#include "logtime_for_MPI.h"
#include "utils_train.h"
#include<mpi.h>
#include "mpi_global_vars.h"
#include <omp.h>
#include "papi.h"

#define L1_SIZE_KB 32
#define L2_SIZE_KB 256
#define L3_SIZE_KB 40960


using namespace std;
typedef double Real;

RegressorTrain::RegressorTrain(const TrainingParameters &tp)
	: training_parameters_(tp)
{
	ferns_ = vector<FernTrain>(training_parameters_.K, FernTrain(tp)); // create a vector of FernTrain objects
	pixels_ = std::vector<std::pair<int, cv::Point2d>>(training_parameters_.P); // create a vector of pairs of integers and cv::Point2d
}

void RegressorTrain::Regress(const vector<cv::Point2d> &mean_shape, // Performs regression on a set of training data using a set of landmarks (mean_shape) and targets.
	vector<vector<cv::Point2d>> *targets,
	const vector<DataPoint> & training_data)
{    //select training_parameters_.P number of features
	for (int i = 0; i < training_parameters_.P; ++i) // create random pixel offsets and store them in the pixels_ vector
	{
		pixels_[i].first = cv::theRNG().uniform(0, 
			training_data[0].landmarks.size()); // uniform distribution between 0 and number of landmarks in the first of first element of training data
		pixels_[i].second.x = cv::theRNG().uniform(-training_parameters_.Kappa, 
			training_parameters_.Kappa);
		pixels_[i].second.y = cv::theRNG().uniform(-training_parameters_.Kappa, 
			training_parameters_.Kappa);
	}
	// If you want to use AVX2, you must pay attention to memory alignment.
	// AVX2 is not used by default. You can change Covariance in fern_train.cpp
	// to AVXCovariance to enable it.
	unique_ptr<double[]> pixels_val_data(new double[   // create pixels_val, a cv::Mat to hold the values of each pixel offset for each training image
		training_parameters_.P * training_data.size() + 3]);

	cv::Mat pixels_val(training_parameters_.P, training_data.size(), CV_64FC1, // Create a 2-D matrix called "pixels_val" with size P x training_data.size() and entry data type double.
	cv::alignPtr(pixels_val_data.get(), 32));

	// Lucy's changes ---------------------------------------------------------------------------------------- // 
	MPI_Barrier(MPI_COMM_WORLD);
	
	const double begin_time = MPI_Wtime(); 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int start = rank * (pixels_val.cols/ size);
    int end = (rank == size - 1) ? pixels_val.cols : (rank + 1) * (pixels_val.cols / size);

	for (int i = start; i < end; ++i) // fill in pixels_val with the values of each pixel offset for each training image
	{
		Transform t = Procrustes(training_data[i].init_shape, mean_shape); // calculate the transformation from the initial shape to the mean shape
		vector<cv::Point2d> offsets(training_parameters_.P); // apply the transformation to each pixel offset
		for (int j = 0; j < training_parameters_.P; ++j) // find the pixel value for each pixel offset and store it in pixels_val
			offsets[j] = pixels_[j].second;
		t.Apply(&offsets, false);
		// For each pixel point, retrieve the pixel value from the image and populate pixels_val with it.


		#pragma omp parallel for schedule(static)
		for (int j = 0; j < training_parameters_.P; ++j)
		{
			
			cv::Point pixel_pos = training_data[i].init_shape[pixels_[j].first]  // Calculate the pixel position with offset in the image
				+ offsets[j];
			if (pixel_pos.inside(cv::Rect(0, 0,    // Check if the pixel position is inside the face rectangle
				training_data[i].image.cols, training_data[i].image.rows)))
			{
				pixels_val.at<double>(j, i) =    // Store the pixel value in pixels_val matrix
					training_data[i].image.at<uchar>(pixel_pos);
			}
			else
				pixels_val.at<double>(j, i) = 0;  // If the pixel position is outside the image, set the pixel value to 0
		}
	}


	// Lucy's changes end ------------------------------------------------------------------------ // 


	cv::Mat pixels_cov, means;  // Create matrices for covariance and means calculation

	MPI_Barrier(MPI_COMM_WORLD);
	const double end_time = MPI_Wtime(); 
	logtime_for_MPI(end_time-begin_time, "Regress-barrier");

	cv::calcCovarMatrix(pixels_val, pixels_cov, means,  // Calculate the covariance matrix and means vector of pixels_val matrix
		cv::COVAR_NORMAL | cv::COVAR_SCALE | cv::COVAR_COLS);
	// Loop through all ferns and regress the targets based on pixels_val matrix and its covariance

	
	for (int i = 0; i < training_parameters_.K; ++i)
	{
		ferns_[i].Regress(targets, pixels_val, pixels_cov);
		for (int j = 0; j < targets->size(); ++j) // Loop through all targets and update the target shape based on the fern output
		{
			(*targets)[j] = ShapeDifference((*targets)[j], ferns_[i].Apply(
				pixels_val(cv::Range::all(), cv::Range(j, j + 1))));
		}
	}

	CompressFerns_Test();//compresses the ferns, which are decision trees with binary splits used for regression.
}




void RegressorTrain::CompressFerns_Test() // This function compresses the ferns by selecting a smaller base of outputs and finding the indices of the fern outputs that are closest to the base using orthogonal matching pursuit (OMP).
{


	int q[] = {5, 10, 20, 30, 40, 50};

	int length = sizeof(q) / sizeof(q[0]);

	for (int q_id = 0; q_id < length; q_id++) {
		base_.create(q[q_id] * 2, training_parameters_.Base, CV_64FC1);// Create a matrix for the base and initialize a vector of random indices.
	vector<int> rand_index; // Define a vector of indices for random index shuffling

	// Lucy's changes ---------------------------------------------------------------------------------------- // 

	for (int i = 0; i < training_parameters_.K * (1 << training_parameters_.F); ++i)
		rand_index.push_back(i);
	random_shuffle(rand_index.begin(), rand_index.end()); // Randomly shuffle the indices and select the first training_parameters_.Base indices to use as the base.

	for (int i = 0; i < training_parameters_.Base; ++i) // Create the base matrix using the shuffled indices
	{
		const vector<cv::Point2d> &output = ferns_[rand_index[i] >> training_parameters_.F]
			.outputs[rand_index[i] & ((1 << training_parameters_.F) - 1)];
		for (int j = 0; j < output.size(); ++j) // Add each x and y value to the corresponding row in the base matrix
		{
			base_.at<double>(j * 2, i) = output[j].x;
			base_.at<double>(j * 2 + 1, i) = output[j].y;
		}
		cv::normalize(base_.col(i), base_.col(i)); // Normalize the column of the base matrix
	}


	for (int i = 0; i < 1; ++i) // For each fern, find the indices of the closest outputs to the base using OMP.
	{
		for (int j = 0; j < 1; ++j)
		{
			const vector<cv::Point2d> &output = ferns_[i].outputs[j];
			cv::Mat output_mat(base_.rows, 1, CV_64FC1); // Create a matrix for each output
			for (int k = 0; k < output.size(); ++k) 
			{// Populate the matrix with x and y values
				output_mat.at<double>(k * 2) = output[k].x;
				output_mat.at<double>(k * 2 + 1) = output[k].y;
			}


			int event_set = PAPI_NULL;
			int events[5] = {PAPI_TOT_CYC,
							PAPI_TOT_INS,
							PAPI_LST_INS,
							PAPI_L1_DCM,
							};


			long long int counters[5];

			int Q = q[q_id];

			std::cout << "======= New Iteration Starts ========" << '\n';
			std::cout << "The parameter Q for this iteration is:" << '\n';
			std::cout << Q << '\n';
			std::cout << '\n';


			PAPI_library_init(PAPI_VER_CURRENT);
			PAPI_create_eventset(&event_set);
			PAPI_add_events(event_set, events, 5);
			PAPI_start(event_set);
			const long long int t0 = PAPI_get_real_nsec();


			double result_2 = OMP_Eigen(output_mat, base_, Q);
			const long long int t1 = PAPI_get_real_nsec();	
			PAPI_stop(event_set, counters);


			const long long total_cycles = counters[0];       // cpu cycles
			const long long total_instructions = counters[1]; // any
			const long long total_load_stores = counters[2];  // number of such instructions
			const long long total_l1d_misses = counters[3];   // number of access request to cache line

			int N = output_mat.rows;  // Get the number of rows
			int M = base_.cols;


			int mem_ops = (N + N * M + 1) + Q * M * N + 2 * Q * M + Q * (Q + 1) + N * N + Q * N;


			const size_t flops = (2 * N - 1) * M * Q + 2 + M * Q + (2 * N - 1)* ( Q / 6) * (Q + 1) * (2 * Q + 1) + Q * (Q + 1) /2 * (2 * N - 1) + 13 * 0.25 * (Q * Q * Q * Q + 2 * Q * Q * Q + Q * Q) + Q * (Q + 1) / 2 * (2 * N - 1) + N * Q;
			// const size_t mem_ops = 3 * n;
			const double twall = (static_cast<double>(t1) - t0) * 1.0e-9; // seconds
			const double IPC = static_cast<double>(total_instructions) / total_cycles;
			const double OI =
				static_cast<double>(flops) / (total_load_stores * sizeof(Real));
			const double OI_theory =
				static_cast<double>(flops) / (mem_ops * sizeof(Real));
			const double float_perf = flops / twall * 1.0e-9; // Gflop/s


			std::cout << "Performance Analysis Eigen library" << '\n';
			std::cout << "Total cycles:                 " << total_cycles << '\n';
			std::cout << "Total instructions:           " << total_instructions << '\n';
			std::cout << "Instructions per cycle (IPC): " << IPC << '\n';
			std::cout << "L1 cache size:                " << L1_SIZE_KB << " KB\n";
			std::cout << "L2 cache size:                " << L2_SIZE_KB << " KB\n";
			std::cout << "L3 cache size:                " << L3_SIZE_KB << " KB\n";
			// std::cout << "Total problem size:           " << 2 * n * sizeof(Real) / 1024
					// << " KB\n";
			std::cout << "Total L1 data misses:         " << total_l1d_misses << '\n';
			std::cout << "Total load/store:             " << total_load_stores << '\n' << " (expected: " << mem_ops << ")\n";
			std::cout << "Operational intensity:        " << std::scientific << OI << '\n' << " (expected: " << OI_theory << ")\n";
			std::cout << "Performance [Gflop/s]:        " << float_perf << '\n';
			std::cout << "Wall-time   [micro-seconds]:  " << twall * 1.0e6 << '\n';
			std::cout << '\n';

			/* ========================================================================================================= */
			PAPI_reset(4);
			PAPI_shutdown();

			int event_set_2 = PAPI_NULL;
			int events_2[5] = {PAPI_TOT_CYC,
							PAPI_TOT_INS,
							PAPI_LST_INS,
							PAPI_L1_DCM,
							};


			long long int counters_2[5];
			
			PAPI_library_init(PAPI_VER_CURRENT);
			PAPI_create_eventset(&event_set_2);
			PAPI_add_events(event_set_2, events_2, 5);
			PAPI_start(event_set_2);
			const long long int t2 = PAPI_get_real_nsec();
			double result = OMP(output_mat, base_, Q);
			const long long int t3 = PAPI_get_real_nsec();	
			PAPI_stop(event_set_2, counters_2);


			const long long total_cycles_2 = counters_2[0];       // cpu cycles
			const long long total_instructions_2 = counters_2[1]; // any
			const long long total_load_stores_2 = counters_2[2];  // number of such instructions
			const long long total_l1d_misses_2 = counters_2[3];   // number of access request to cache line



			const size_t flops_2 = (2 * N - 1) * M * Q + 2 + M * Q + (2 * N - 1)* ( Q / 6) * (Q + 1) * (2 * Q + 1) + Q * (Q + 1) /2 * (2 * N - 1) + 13 * 0.25 * (Q * Q * Q * Q + 2 * Q * Q * Q + Q * Q) + Q * (Q + 1) / 2 * (2 * N - 1) + N * Q;
			// const size_t mem_ops = 3 * n;
			const double twall_2 = (static_cast<double>(t3) - t2) * 1.0e-9; // seconds
			const double IPC_2 = static_cast<double>(total_instructions_2) / total_cycles_2;
			const double OI_2 =
				static_cast<double>(flops_2) / (total_load_stores_2 * sizeof(Real));
			// const double OI_theory =
				// static_cast<double>(flops) / (mem_ops * sizeof(Real));
			const double float_perf_2 = flops_2 / twall_2 * 1.0e-9; // Gflop/s


			std::cout << "Performance Analysis original Library " << '\n';
			std::cout << "Total cycles:                 " << total_cycles_2 << '\n';
			std::cout << "Total instructions:           " << total_instructions_2 << '\n';
			std::cout << "Instructions per cycle (IPC): " << IPC_2 << '\n';
			std::cout << "L1 cache size:                " << L1_SIZE_KB << " KB\n";
			std::cout << "L2 cache size:                " << L2_SIZE_KB << " KB\n";
			std::cout << "L3 cache size:                " << L3_SIZE_KB << " KB\n";
			// std::cout << "Total problem size:           " << 2 * n * sizeof(Real) / 1024
					// << " KB\n";
			std::cout << "Total L1 data misses:         " << total_l1d_misses_2 << '\n';
			std::cout << "Total load/store:             " << total_load_stores_2 << '\n';
					// << " (expected: " << mem_ops << ")\n";
			std::cout << "Operational intensity:        " << std::scientific << OI_2 << '\n';
					// << " (expected: " << OI_theory << ")\n";
			std::cout << "Performance [Gflop/s]:        " << float_perf_2 << '\n';
			std::cout << "Wall-time   [micro-seconds]:  " << twall_2 * 1.0e6 << '\n';


			std::cout << '\n';

			std::cout << "The difference in result is small at";
			std::cout << abs(result - result_2);
			std::cout << '\n';


			}
		}
	}
}
