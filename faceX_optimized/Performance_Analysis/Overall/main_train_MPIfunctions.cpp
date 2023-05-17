#include "meanshapeMPI.h"
#include "logtime_for_MPI.h"
#include "main_train_MPIfunctions.h"
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <map>
#include <algorithm>
#include <numeric>
#include <stdexcept>


#include <opencv2/opencv.hpp>

#include<mpi.h>
#include "mpi_global_vars.h"
#include <omp.h>



using namespace std;

int training_image_cnt;

// Serialize DataPoint structure into a vector of chars
void serializeDataPoint(const DataPoint &dp, std::vector<char> &buffer)
{
    // Extract relevant information from the DataPoint structure
    int rows = dp.image.rows;
    int cols = dp.image.cols;
    int type = dp.image.type();
    int face_rect_data[4] = {dp.face_rect.x, dp.face_rect.y, dp.face_rect.width, dp.face_rect.height};
    int landmarks_size = dp.landmarks.size();
    int init_shape_size = dp.init_shape.size();

    // Insert each data element into the buffer
    buffer.insert(buffer.end(), reinterpret_cast<char*>(&rows), reinterpret_cast<char*>(&rows) + sizeof(int));
    buffer.insert(buffer.end(), reinterpret_cast<char*>(&cols), reinterpret_cast<char*>(&cols) + sizeof(int));
    buffer.insert(buffer.end(), reinterpret_cast<char*>(&type), reinterpret_cast<char*>(&type) + sizeof(int));
    buffer.insert(buffer.end(), reinterpret_cast<char*>(dp.image.data), reinterpret_cast<char*>(dp.image.data) + rows * cols * dp.image.elemSize());
    buffer.insert(buffer.end(), reinterpret_cast<char*>(face_rect_data), reinterpret_cast<char*>(face_rect_data) + 4 * sizeof(int));
    buffer.insert(buffer.end(), reinterpret_cast<char*>(&landmarks_size), reinterpret_cast<char*>(&landmarks_size) + sizeof(int));

    // Insert the landmarks and init_shape points into the buffer
    for (const cv::Point2d &p : dp.landmarks)
    {
        buffer.insert(buffer.end(), reinterpret_cast<const char*>(&p.x), reinterpret_cast<const char*>(&p.x) + sizeof(double));
        buffer.insert(buffer.end(), reinterpret_cast<const char*>(&p.y), reinterpret_cast<const char*>(&p.y) + sizeof(double));
    }

    buffer.insert(buffer.end(), reinterpret_cast<char*>(&init_shape_size), reinterpret_cast<char*>(&init_shape_size) + sizeof(int));

    for (const cv::Point2d &p : dp.init_shape)
    {
        buffer.insert(buffer.end(), reinterpret_cast<const char*>(&p.x), reinterpret_cast<const char*>(&p.x) + sizeof(double));
        buffer.insert(buffer.end(), reinterpret_cast<const char*>(&p.y), reinterpret_cast<const char*>(&p.y) + sizeof(double));
    }
}

// Deserialize a DataPoint structure from a vector of chars
DataPoint deserializeDataPoint(const std::vector<char> &buffer, size_t &offset)
{
    // Initialize a DataPoint structure and variables to hold deserialized data
    DataPoint dp;
    int rows, cols, type;
    int face_rect_data[4];
    int landmarks_size;
    int init_shape_size;

    // Deserialize data from the buffer and update the offset
    std::memcpy(&rows, &buffer[offset], sizeof(int));
    offset += sizeof(int);
    std::memcpy(&cols, &buffer[offset], sizeof(int));
    offset += sizeof(int);
    std::memcpy(&type, &buffer[offset], sizeof(int));
    offset += sizeof(int);

    dp.image.create(rows, cols, type);
    std::memcpy(dp.image.data, &buffer[offset], rows * cols * dp.image.elemSize());
    offset += rows * cols * dp.image.elemSize();

    // Create the face rectangle from the deserialized data
    std::memcpy(face_rect_data, &buffer[offset], 4 * sizeof(int));
    offset += 4 * sizeof(int);
    dp.face_rect = cv::Rect(face_rect_data[0], face_rect_data[1], face_rect_data[2], face_rect_data[3]);

    // Deserialize the landmarks and store them in the DataPoint structure
    std::memcpy(&landmarks_size, &buffer[offset], sizeof(int));
    offset += sizeof(int);

    for (int i = 0; i < landmarks_size; ++i)
    {
        cv::Point2d p;
        std::memcpy(&p.x, &buffer[offset], sizeof(double));
        offset += sizeof(double);
        std::memcpy(&p.y, &buffer[offset], sizeof(double));
        offset += sizeof(double);
        dp.landmarks.push_back(p);
    }
    
    // Deserialize the init_shape points and store them in the DataPoint structure
    std::memcpy(&init_shape_size, &buffer[offset], sizeof(int));
    offset += sizeof(int);
    
    for (int i = 0; i < init_shape_size; ++i)
    {
        cv::Point2d p;
        std::memcpy(&p.x, &buffer[offset], sizeof(double));
        offset += sizeof(double);
        std::memcpy(&p.y, &buffer[offset], sizeof(double));
        offset += sizeof(double);
        dp.init_shape.push_back(p);
    }
    
    return dp;
}


// function to generate augmented data points using MPI
vector<DataPoint> ArgumentData_MPI(const vector<DataPoint> &training_data, int factor)
{
    // Record the start time of the function for performance logging
    const double begin_time = MPI_Wtime(); 
    
    // Validate input arguments
    if (training_data.size() < 2 * factor)
    {
        throw invalid_argument("You should provide training data with at least "
                               "2*ArgumentDataFactor images.");
    }

    // Calculate the start and end indices of the training data for each MPI process
    int size = training_data.size();
    int chunk_size = size / process_cnt;
    int start_idx = process_rank * chunk_size;
    int end_idx = (process_rank == process_cnt - 1) ? size : start_idx + chunk_size;
    const int local_cnt = end_idx - start_idx ;
    
    //loop over the start and end as specified above
    vector<DataPoint> result(local_cnt * factor);
    for (int i = start_idx; i < end_idx; ++i)
    {
        set<int> shape_indices;
        while (shape_indices.size() < factor)
        {
            int rand_index = cv::theRNG().uniform(0, size);
            if (rand_index != i)
                shape_indices.insert(rand_index);
        }

        auto it = shape_indices.cbegin();
        // augment data based on a given factor in the function input parameters
        for (int j = (i - start_idx) * factor; j < (i - start_idx + 1) * factor; ++j, ++it)
        {
            result[j] = training_data[i];
            result[j].init_shape = MapShape(training_data[*it].face_rect, 
                                            training_data[*it].landmarks, result[j].face_rect);
        }
    }
    /*
	{
	    string filename = "main_train_ArgumentData_testresultloop_MPI_rank"+ to_string(process_rank) +".yml";
        cv::FileStorage file;
        file.open(filename, cv::FileStorage::WRITE);
        for (int i = 0; i < result.size(); i++){
            file << "image" << result[i].image;
            file << "rect" << result[i].face_rect;
            file << "init_shape" << result[i].init_shape;
            file << "landmarks" << result[i].landmarks;
        }
        file.release();
	}    
    */
    // pack each process's local DataPoints into a serialized data array
    std::vector<char> send_buffer_dp;
    for (const DataPoint &dp : result)
    {
        serializeDataPoint(dp, send_buffer_dp);
    }
    
    // Gather the sizes of each process's send buffer
    int send_count = send_buffer_dp.size();
    std::vector<int> recv_counts(process_cnt);

    MPI_Gather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate displacements for MPI_Gatherv
    std::vector<int> displs(process_cnt, 0);
    for (int i = 1; i < process_cnt; ++i)
    {
        displs[i] = displs[i - 1] + recv_counts[i - 1];
    }
    
    // Gather the serialized data at process_rank 0
    std::vector<char> recv_buffer_dp;
    if (process_rank == 0)
    {
        recv_buffer_dp.resize(displs.back() + recv_counts.back());
    }

    MPI_Gatherv(send_buffer_dp.data(), send_count, MPI_CHAR, recv_buffer_dp.data(), recv_counts.data(), displs.data(), MPI_CHAR, 0, MPI_COMM_WORLD);

    // Deserialize the received data and merge it into the complete result vector
    if (process_rank == 0)
    {
        result.clear();
        size_t offset_dp = 0;
        while (offset_dp < recv_buffer_dp.size())
        {
            result.push_back(deserializeDataPoint(recv_buffer_dp, offset_dp));
        }
    }
    
    // Broadcast the size of the complete result vector
    int complete_result_size = 0;
    if (process_rank == 0)
    {
        complete_result_size = result.size();
    }

    MPI_Bcast(&complete_result_size, 1, MPI_INT, 0, MPI_COMM_WORLD);    

    // Serialize the complete result vector on process_rank 0
    std::vector<char> complete_result_buffer;
    if (process_rank == 0)
    {
        for (const DataPoint &dp : result)
        {
            serializeDataPoint(dp, complete_result_buffer);
        }
    }
    
    // Broadcast the size of the complete result buffer
    int complete_result_buffer_size = 0;
    if (process_rank == 0)
    {
        complete_result_buffer_size = complete_result_buffer.size();
    }

    MPI_Bcast(&complete_result_buffer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize the receive buffer on all ranks and broadcast the complete result buffer
    if (process_rank != 0)   {
        complete_result_buffer.resize(complete_result_buffer_size);
    }
    //cout<< "before MPI_Bcast full data" <<endl;
    MPI_Bcast(complete_result_buffer.data(), complete_result_buffer_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    //cout<< "after MPI_Bcast full data" <<endl;
    if (process_rank != 0)
    {
        size_t offset_d = 0;
        result.clear();
        while (offset_d < complete_result_buffer.size())
        {
            result.push_back(deserializeDataPoint(complete_result_buffer, offset_d));
        }
    }
    /*
	{
	    string filename = "main_train_ArgumentData_MPI_rank"+ to_string(process_rank) +".yml";
        cv::FileStorage file;
        file.open(filename, cv::FileStorage::WRITE);
        for (int i = 0; i < result.size(); i++){
            file << "image" << result[i].image;
            file << "rect" << result[i].face_rect;
            file << "init_shape" << result[i].init_shape;
            file << "landmarks" << result[i].landmarks;
        }
        file.release();
	}
	*/
	// compute and log function processing time
    const double end_time = MPI_Wtime(); 
    logtime_for_MPI(end_time-begin_time, "ArgumentData_MPI");
    return result;
}


// Create initial shapes for testing using MPI and k-means clustering
vector<vector<cv::Point2d>> CreateTestInitShapes_MPI( const vector<DataPoint>& training_data, const TrainingParameters& tp) 
{
    // Start the timer to measure the function's processing time
    const double begin_time = MPI_Wtime(); 
    
    // Check if the number of test initial shapes is larger than the training image count, throw an exception if true
    if (tp.TestInitShapeCount > training_data.size())
    {
        throw invalid_argument("TestInitShapeCount is larger than training image count"
                               ", which is not allowed.");
    }
    // Get the size of landmarks in the training data
    const int kLandmarksSize = training_data[0].landmarks.size();

    // Get the size of the training data and calculate chunk size for each MPI process
    const int size = training_data.size();
    const int chunk_size = size / process_cnt;
    const int start_idx = process_rank * chunk_size;
    const int end_idx = (process_rank == process_cnt - 1) ? size : start_idx + chunk_size;

    // Create a matrix to store landmarks for each MPI process
    cv::Mat rank_landmarks(end_idx - start_idx, kLandmarksSize * 2, CV_32FC1);
    
    // Iterate through only the corresponding start and end of the full training data for each process
    for (int i = start_idx; i < end_idx; ++i)
    {
        // Map landmarks to a unit rectangle
        vector<cv::Point2d> landmarks = MapShape(training_data[i].face_rect,
                                                 training_data[i].landmarks, cv::Rect(0, 0, 1, 1));
        // Store the mapped landmarks in the rank_landmarks matrix                                         
        for (int j = 0; j < kLandmarksSize; ++j)
        {
            rank_landmarks.at<float>(i - start_idx, j * 2) = static_cast<float>(landmarks[j].x);
            rank_landmarks.at<float>(i - start_idx, j * 2 + 1) = static_cast<float>(landmarks[j].y);
        }
    }
    
    // Prepare buffer and displacement for Allgatherv
    vector<int> recv_counts(process_cnt, chunk_size * kLandmarksSize * 2);
    recv_counts[process_cnt - 1] = (size - (process_cnt - 1) * chunk_size) * kLandmarksSize * 2;
    vector<int> displs(process_cnt, 0);
    for (int i = 1; i < process_cnt; ++i)
    {
        displs[i] = displs[i - 1] + recv_counts[i - 1];
    }

    cv::Mat all_landmarks(size, kLandmarksSize * 2, CV_32FC1);
    MPI_Allgatherv(rank_landmarks.data, rank_landmarks.rows * rank_landmarks.cols, MPI_FLOAT,
                   all_landmarks.data, recv_counts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);

    // Perform k-means clustering on the all_landmarks matrix
    cv::Mat labels, centers;
    //cv::setRNGSeed(42);
    cv::kmeans(all_landmarks, tp.TestInitShapeCount, labels,
               cv::TermCriteria(cv::TermCriteria::COUNT, 50, 0),
               10, cv::KMEANS_RANDOM_CENTERS | cv::KMEANS_PP_CENTERS, centers);
    
    // Initialize a vector to store the resulting initial shapes
    vector<vector<cv::Point2d>> result;
    
    // Iterate through the cluster centers
    for (int i = 0; i < tp.TestInitShapeCount; ++i)
    {
        // Initialize a vector to store the landmarks of the current initial shape
        vector<cv::Point2d> landmarks;
        
        // Iterate through the landmarks of the current initial shape
        for (int j = 0; j < kLandmarksSize; ++j)
        {
            // Get the coordinates of the current landmark and store it in the landmarks vector
            landmarks.push_back(cv::Point2d(
                centers.at<float>(i, j * 2), centers.at<float>(i, j * 2 + 1)));
        }
        result.push_back(landmarks);
    }
    
    // Uncomment the following lines to save the resulting initial shapes to a file
    /*
    {
        string filename = "main_train_test_init_shapes_MPI_rank"+to_string(process_rank)+".yml";
        cv::FileStorage file;
        file.open(filename, cv::FileStorage::WRITE);
        file << "test_init_shapes" << result;
        file.release();
    }  
    */
    // compute and log function processing time
    const double end_time = MPI_Wtime(); 
    logtime_for_MPI(end_time-begin_time, "CreateTestInitShapes_MPI");
    return result;
}



// multiprocess read of training images
vector<DataPoint> GetTrainingData_MPI(const TrainingParameters &tp)
{
    // record beginning time of the function
    //const double begin_time = MPI_Wtime(); 
    
    // Read the label file
    const string label_pathname = tp.training_data_root + "/labels.txt";
    ifstream fin(label_pathname);
    if (!fin)
        throw runtime_error("Cannot open label file " + label_pathname + " (Pay attention to path separator!)");

    vector<DataPoint> result;
    string current_image_pathname;

    // Get all lines in the label file
    int total_line_cnt = 0;
    vector<string> lines;
    string line;
    while (getline(fin, line)){
        ++total_line_cnt;
        lines.push_back(line);
    }
    fin.close();         

    // Find the corresponding lines in labels that each ranks hould handle
    int process_line_cnt = total_line_cnt / process_cnt; // number of lines to read in by each process
    int start_line_idx = process_rank  * process_line_cnt;
    int end_line_idx = (process_rank  == process_cnt - 1) ? total_line_cnt : start_line_idx + process_line_cnt;
    const int rank_img_cnt = end_line_idx - start_line_idx ;

    for (int line_idx = start_line_idx; line_idx < end_line_idx; ++line_idx)
    {
        // Extract the image filename and load the image
        stringstream ss(lines[line_idx]);
        ss >> current_image_pathname;
        
        DataPoint current_data_point;
        
        current_data_point.image = cv::imread(tp.training_data_root + "/" + current_image_pathname, CV_LOAD_IMAGE_GRAYSCALE);
        if (current_data_point.image.data == nullptr)
            throw runtime_error("Cannot open image file " + current_image_pathname + " (Pay attention to path separator!)");
            
        // Read the face rectangle and landmarks
        int left, right, top, bottom;
        ss >> left >> right >> top >> bottom;
        current_data_point.face_rect = cv::Rect(left, top, right - left + 1, bottom - top + 1);
        for (int i = 0; i < tp.landmark_count; ++i)
        {
            cv::Point2d p;
            ss >> p.x >> p.y;
            current_data_point.landmarks.push_back(p);
        }
        result.push_back(current_data_point);
    }

    // pack each process's local DataPoints into a serialized data array
    std::vector<char> send_buffer_dp;
    for (const DataPoint &dp : result)
    {
        serializeDataPoint(dp, send_buffer_dp);
    }
    
    // Gather the sizes of each process's send buffer
    int send_count = send_buffer_dp.size();
    std::vector<int> recv_counts(process_cnt);
    MPI_Gather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate displacements for MPI_Gatherv
    std::vector<int> displs(process_cnt, 0);
    for (int i = 1; i < process_cnt; ++i)
    {
        displs[i] = displs[i - 1] + recv_counts[i - 1];
    }
    
    // Gather the serialized data at process_rank 0
    std::vector<char> recv_buffer_dp;
    if (process_rank == 0)
    {
        recv_buffer_dp.resize(displs.back() + recv_counts.back());
    }
    MPI_Gatherv(send_buffer_dp.data(), send_count, MPI_CHAR, recv_buffer_dp.data(), recv_counts.data(), displs.data(), MPI_CHAR, 0, MPI_COMM_WORLD);
    
    // Deserialize the received data and merge it into the complete result vector
    if (process_rank == 0)
    {
        result.clear();
        size_t offset_dp = 0;
        while (offset_dp < recv_buffer_dp.size())
        {
            result.push_back(deserializeDataPoint(recv_buffer_dp, offset_dp));
        }
    }
    /*
    {
        string filename = "main_train_MPIgatherv_rank"+to_string(process_rank)+"result.yml";
        cv::FileStorage file;
        file.open(filename, cv::FileStorage::WRITE);
        for (int i = 0; i < result.size(); i++){
            file << "image" << result[i].image;
            file << "rect" << result[i].face_rect;
            file << "init_shape" << result[i].init_shape;
            file << "landmarks" << result[i].landmarks;
        }
        file.release();
    }    
    */
    // Broadcast the size of the complete result vector
    int complete_result_size = 0;
    if (process_rank == 0)
    {
        complete_result_size = result.size();
    }
    MPI_Bcast(&complete_result_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Serialize the complete result vector on process_rank 0
    std::vector<char> complete_result_buffer;
    if (process_rank == 0)
    {
        for (const DataPoint &dp : result)
        {
            serializeDataPoint(dp, complete_result_buffer);
        }
    }
    
    // Broadcast the size of the complete result buffer
    int complete_result_buffer_size = 0;
    if (process_rank == 0)
    {
        complete_result_buffer_size = complete_result_buffer.size();
    }
    MPI_Bcast(&complete_result_buffer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Resize the receive buffer on all ranks and broadcast the complete result buffer
    if (process_rank != 0)   {
        complete_result_buffer.resize(complete_result_buffer_size);
    }

    MPI_Bcast(complete_result_buffer.data(), complete_result_buffer_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // update each rank's DataPoint to have the full image data
    if (process_rank != 0)
    {
        size_t offset_d = 0;
        result.clear();
        while (offset_d < complete_result_buffer.size())
        {
            result.push_back(deserializeDataPoint(complete_result_buffer, offset_d));
        }
    }
    /*
    {
        string filename = "main_train_MPIgatherv_rank"+to_string(process_rank)+"finalresult.yml";
        cv::FileStorage file;
        file.open(filename, cv::FileStorage::WRITE);
        for (int i = 0; i < result.size(); i++){
            file << "image" << result[i].image;
            file << "rect" << result[i].face_rect;
            file << "init_shape" << result[i].init_shape;
            file << "landmarks" << result[i].landmarks;
        }
        file.release();
    }    
    */
    // compute and log function processing time
    //training_image_cnt = result.size();
    //const double end_time = MPI_Wtime(); 
    //logtime_for_MPI(end_time-begin_time, "GetTrainingData_MPI");
    return result;
}
