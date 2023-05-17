
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
#include "utils_train.h"
#include "logtime_for_MPI.h"
#include<mpi.h>
#include "mpi_global_vars.h"
#include <omp.h>
using namespace std;


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

	// Write data to the storage object
	/*
	if (process_rank==0){
	    string filename = "training_data.yml";
	    cv::FileStorage file;
	    file.open(filename, cv::FileStorage::WRITE);
	    for (int i = 0; i < training_data.size(); i++){
          file << "image" << training_data[i].image;
          file << "rect" << training_data[i].face_rect;
          file << "init_shape" << training_data[i].init_shape;
          file << "landmarks" << training_data[i].landmarks;
         }
	    file.release();
	}
	*/

	// If you want to use AVX2, you must pay attention to memory alignment.
	// AVX2 is not used by default. You can change Covariance in fern_train.cpp
	// to AVXCovariance to enable it.
	unique_ptr<double[]> pixels_val_data(new double[   // create pixels_val, a cv::Mat to hold the values of each pixel offset for each training image
		training_parameters_.P * training_data.size() + 3]);

	cv::Mat pixels_val(training_parameters_.P, training_data.size(), CV_64FC1, // Create a 2-D matrix called "pixels_val" with size P x training_data.size() and entry data type double.
	cv::alignPtr(pixels_val_data.get(), 32));

	// MPI Implementation -------------------------------------------------------------------------------- // 
	MPI_Barrier(MPI_COMM_WORLD);
	
//	const double begin_time = MPI_Wtime(); 
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


	// --------------------------------------------------------------------------------------------------- // 


	cv::Mat pixels_cov, means;  // Create matrices for covariance and means calculation

	MPI_Barrier(MPI_COMM_WORLD);
//	const double end_time = MPI_Wtime(); 
//	logtime_for_MPI(end_time-begin_time, "Regress-barrier");

	cv::calcCovarMatrix(pixels_val, pixels_cov, means,  // Calculate the covariance matrix and means vector of pixels_val matrix
		cv::COVAR_NORMAL | cv::COVAR_SCALE | cv::COVAR_COLS);
	// Loop through all ferns and regress the targets based on pixels_val matrix and its covariance

	
	for (int i = 0; i < training_parameters_.K; ++i)
	{
		ferns_[i].Regress(targets, pixels_val, pixels_cov);
		for (int j = 0; j < targets->size(); ++j) // Loop through all targets and update the target shape based on the fern output
		{
			(*targets)[j] = ShapeDifference((*targets)[j], ferns_[i].Apply( //����ÿ��ShapeDifference���ص�targets elementҪô��targets[j]Ҫô��0
				pixels_val(cv::Range::all(), cv::Range(j, j + 1))));
		}
	}

	CompressFerns();//compresses the ferns, which are decision trees with binary splits used for regression.
}

void RegressorTrain::CompressFerns() // This function compresses the ferns by selecting a smaller base of outputs and finding the indices of the fern outputs that are closest to the base using orthogonal matching pursuit (OMP).
{
	base_.create(ferns_[0].outputs[0].size() * 2, training_parameters_.Base, CV_64FC1);// Create a matrix for the base and initialize a vector of random indices.
	vector<int> rand_index; // Define a vector of indices for random index shuffling
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < training_parameters_.K * (1 << training_parameters_.F); ++i)
		rand_index.push_back(i);
	random_shuffle(rand_index.begin(), rand_index.end()); // Randomly shuffle the indices and select the first training_parameters_.Base indices to use as the base.

	#pragma omp parallel for schedule(static)
	for (int i = 0; i < training_parameters_.Base ; ++i) // Create the base matrix using the shuffled indices
	{
		const int idx = (rand_index[i] >> training_parameters_.F) % training_parameters_.K;
		const vector<cv::Point2d> &output = ferns_[idx]
			.outputs[rand_index[i] & ((1 << training_parameters_.F) - 1)];
		for (int j = 0; j < output.size(); ++j) // Add each x and y value to the corresponding row in the base matrix
		{
			base_.at<double>(j * 2, i) = output[j].x;
			base_.at<double>(j * 2 + 1, i) = output[j].y;
		}
		cv::normalize(base_.col(i), base_.col(i)); // Normalize the column of the base matrix
	}
	
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < training_parameters_.K; ++i) // For each fern, find the indices of the closest outputs to the base using OMP.
	{
		for (int j = 0; j < (1 << training_parameters_.F); ++j)
		{
			const vector<cv::Point2d> &output = ferns_[i].outputs[j];
			cv::Mat output_mat(base_.rows, 1, CV_64FC1); // Create a matrix for each output
			for (int k = 0; k < output.size(); ++k) 
			{// Populate the matrix with x and y values
				output_mat.at<double>(k * 2) = output[k].x;
				output_mat.at<double>(k * 2 + 1) = output[k].y;
			}

		// changed to experiment with effect
			ferns_[i].outputs_mini.push_back(OMP_Eigen(output_mat, base_, training_parameters_.Q)); // Compress the output matrix using OMP and the base matrix
		}
	}
	
}

//-----------------------------------CompressFern MPI Implementation-----------------------------//
/*void RegressorTrain::CompressFernsMPI(){
    base_.create(ferns_[0].outputs[0].size() * 2, training_parameters_.Base, CV_64FC1);
    vector<int> rand_index;
    for (int i = 0; i < training_parameters_.K * (1 << training_parameters_.F); ++i)
        rand_index.push_back(i);

    random_shuffle(rand_index.begin(), rand_index.end());

    const double begin_time = MPI_Wtime(); 

    int rank, size, start, end, subset_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    subset_size = ferns_.size() / size;
    start = rank * subset_size;
    end = (rank == size - 1) ? ferns_.size() : (rank + 1) * subset_size;


    for (int i = start; i < end; ++i) {
        for (int j = 0; j < (1 << training_parameters_.F); ++j) {
            const vector<cv::Point2d> &output = ferns_[i].outputs[j];
            cv::Mat output_mat(base_.rows, 1, CV_64FC1);
            for (int k = 0; k < output.size(); ++k) {
                output_mat.at<double>(k * 2) = output[k].x;
                output_mat.at<double>(k * 2 + 1) = output[k].y;
            }
            cv::Mat compressed_output = OMP_Eigen(output_mat, base_, training_parameters_.Q);
            ferns_[i].outputs_mini.push_back(compressed_output);
        }
    }

    // Combine the compressed ferns from all nodes
    for (int node = 0; node < size; ++node) {
        if (rank == node) {
            for (int i = start; i < end; ++i) {
                for (int j = 0; j < (1 << training_parameters_.F); ++j) {
                    MPI_Send(ferns_[i].outputs_mini[j].data, compressed_size * sizeof(double), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
                }
            }
        }
        if (rank == 0) {
            for (int i = start; i < end; ++i) {
                for (int j = 0; j < (1 << training_parameters_.F); ++j) {
                    vector<double> compressed_output(compressed_size);
                    MPI_Recv(compressed_output.data(), compressed_size * sizeof(double), MPI_BYTE, node, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    ferns_[i].outputs_mini[j] = cv::Mat(compressed_size, 1, CV_64FC1, compressed_output.data());
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        CompressFerns();
    }
    const double end_time = MPI_Wtime(); 
	logtime_for_MPI(end_time-begin_time, "CompressFernMPI_Logtime");
}
*///--------------------------------------------Implementation End----------------------------------------//


vector<cv::Point2d> RegressorTrain::Apply(const vector<cv::Point2d> &mean_shape,  // This function applies a trained regression model to a test image.
	const DataPoint &data) const
{
	cv::Mat pixels_val(1, training_parameters_.P, CV_64FC1);// Create a matrix to store the pixel intensities in the image.
	Transform t = Procrustes(data.init_shape, mean_shape); // Align the initial shape of the test image to the mean shape using Procrustes analysis.
	vector<cv::Point2d> offsets(training_parameters_.P); // Compute the offsets of each pixel with respect to the aligned initial shape.
	for (int j = 0; j < training_parameters_.P; ++j)
		offsets[j] = pixels_[j].second;
	t.Apply(&offsets, false);

	double *p = pixels_val.ptr<double>(0);
	for (int j = 0; j < training_parameters_.P; ++j) // Populate the pixels_val matrix with the pixel intensities at the offset positions.
	{
		cv::Point pixel_pos = data.init_shape[pixels_[j].first] + offsets[j];
		if (pixel_pos.inside(cv::Rect(0, 0, data.image.cols, data.image.rows)))
			p[j] = data.image.at<uchar>(pixel_pos);
		else
			p[j] = 0;
	}
	// Apply the trained regression model to the pixels_val matrix and get the output coefficients.
	vector<double> coeffs(training_parameters_.Base);
	for (int i = 0; i < training_parameters_.K; ++i)
		ferns_[i].ApplyMini(pixels_val, coeffs);
	// Compute the final output by taking a linear combination of the basis vectors.
	cv::Mat result_mat = cv::Mat::zeros(mean_shape.size() * 2, 1, CV_64FC1);
	for (int i = 0; i < training_parameters_.Base; ++i)
		result_mat += coeffs[i] * base_.col(i);
	// Convert the output matrix to a vector of 2D points.
	vector<cv::Point2d> result(mean_shape.size());
	for (int i = 0; i < result.size(); ++i)
	{
		result[i].x = result_mat.at<double>(i * 2);
		result[i].y = result_mat.at<double>(i * 2 + 1);
	}
	// Return the final output vector of 2D points.
	return result;
}


void RegressorTrain::write(cv::FileStorage &fs)const
{
	fs << "{";
	fs << "pixels";
	fs << "[";
	for (auto it = pixels_.begin(); it != pixels_.end(); ++it) //loops through the pixels_ map of the RegressorTrain object.
		fs << "{" << "first" << it->first << "second" << it->second << "}";//writes the first and second values of each pixels_ map element to the file storage object as a dictionary with braces.
	fs << "]";
	fs << "ferns" << "[";
	for (auto it = ferns_.begin(); it != ferns_.end(); ++it) //loops through the ferns_ vector of the RegressorTrain object.
		fs << *it;
	fs << "]";
	fs << "base" << base_;
	fs << "}";
}

void write(cv::FileStorage& fs, const string&, const RegressorTrain& r) //writes a RegressorTrain object to a file storage object via the RegressorTrain::write function.
{
	r.write(fs);
}
