#include "meanshapeMPI.h"
#include "logtime_for_MPI.h"
#include "mpi_global_vars.h"
#include <omp.h>
using namespace std;

#define LANDMARKS 5



static void Normalize(vector<cv::Point2d> * shape)
{
    /*
    This function normalizes the input shape by translating it to the origin, 
    scaling it based on the distance between the eyes, and rotating it so that the line between the eyes is horizontal.    
    */

    // Compute the center of the shape
    cv::Point2d center;
    for (const cv::Point2d p : *shape)
        center += p;
    center *= 1.0 / shape->size();

    // Translate the shape to the origin
    for (cv::Point2d &p : *shape)
        p -= center;

    // Define indices for left and right eye landmarks
    const int left_eye_index = 0;
    const int right_eye_index = 1;

    // Extract the coordinates of the left and right eyes
    cv::Point2d left_eye = (*shape).at(left_eye_index);
    cv::Point2d right_eye = (*shape).at(right_eye_index);

    // Calculate the distance between the eyes and compute the scaling factor
    double eyes_distance = cv::norm(left_eye - right_eye);
    double scale = 1 / eyes_distance;

    // Compute the angle of rotation
    double theta = -atan((right_eye.y - left_eye.y) / (right_eye.x - left_eye.x));

    // Create a Transform object for translation and rotation
    Transform t;
    t.scale_rotation(0, 0) = scale * cos(theta);
    t.scale_rotation(0, 1) = -scale * sin(theta);
    t.scale_rotation(1, 0) = scale * sin(theta);
    t.scale_rotation(1, 1) = scale * cos(theta);

    // Apply the transformation to the shape
    t.Apply(shape, false);
}

vector<cv::Point2d> MeanShape_MPIhelper(vector<vector<cv::Point2d>> shapes,double* inital_marks,int total_size)
{
    /*
    This function is a helper function that computes the mean shape of a given set of shapes.  
    */

    const int kIterationCount = 10; // Number of iterations for the mean shape computation
    double reduce_shapes[LANDMARKS * 2];
    vector<cv::Point2d> mean_shape(LANDMARKS);

    // Initialize the mean_shape with the initial landmarks
    for (int i = 0; i < LANDMARKS; i++)
    {
        mean_shape[i].x = inital_marks[2 * i];
        mean_shape[i].y = inital_marks[2 * i+1];
    }

    // Iterate to compute the mean shape
    for (int i = 0; i < kIterationCount; ++i)
    {
        // Align each shape to the current mean shape using Procrustes analysis
        for (vector<cv::Point2d> & shape : shapes)
        {
            Transform t = Procrustes(mean_shape, shape);
            t.Apply(&shape);
        }

        // Reset the mean_shape to the origin
        for (cv::Point2d & p : mean_shape)
            p.x = p.y = 0;

        // Sum the aligned shapes to compute the new mean shape
        for (const vector<cv::Point2d> & shape : shapes)
            for (int j = 0; j < mean_shape.size(); ++j)
            {
                mean_shape[j].x += shape[j].x;
                mean_shape[j].y += shape[j].y;
            }

        // Update the initial marks with the new mean shape
		for (int j = 0; j < LANDMARKS; j++)
		{
			inital_marks[2 * j]=mean_shape[j].x;
			inital_marks[2 * j + 1]= mean_shape[j].y ;
		}
        // Perform an Allreduce operation to sum the partial mean shapes across all processes
        MPI_Allreduce(inital_marks, reduce_shapes, 10, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
        // Update the mean_shape with the reduced values
        for (int j = 0; j < LANDMARKS; j++)
        {
            mean_shape[j].x = reduce_shapes[2 * j];
            mean_shape[j].y = reduce_shapes[2 * j + 1];
        }
    
        // Normalize the mean_shape by dividing by the total number of shapes
        for (cv::Point2d & p : mean_shape)
            p *= 1.0 / total_size;
    
        // Normalize the mean_shape by translating and rotating it
        Normalize(&mean_shape);
	}

	return mean_shape;
}

// calculate overall meanshape assuming that only local rank 0 have full data
std::vector<cv::Point2d> MeanShape_local_MPI(std::vector<std::vector<cv::Point2d>> shapes, const TrainingParameters &tp){
    
    // Record the start time
    const double begin_time = MPI_Wtime(); 
    
	//vector<vector<cv::Point2d>> shapes;
	int shape_size;
	shape_size=shapes.size();

	// Broadcast the shape size to all ranks
	MPI_Bcast(&shape_size, 1, MPI_INT, 0, MPI_COMM_WORLD);


	//compute the size of each rank
	int local_size = shape_size / process_cnt;
	int remain = (shape_size % process_cnt);
	if (process_rank < remain)
		local_size += 1;
	// number of double values for each shape;
	int shape_elements = 10;
	
	// prepare MPI scatter buffer 
	int *sendcounts= new int [process_cnt];
	int *displ=      new int[process_cnt];
	for (int i = 0; i < process_cnt; i++)
	{
		if (i < remain)
		{
			sendcounts[i] = ((shape_size / process_cnt) + 1)*shape_elements;
		}
		else
			sendcounts[i] = (shape_size / process_cnt)*shape_elements;

	}
	displ[0] = 0;
	for (int i = 1; i < process_cnt; i++)
	{
		displ[i] = displ[i - 1] + sendcounts[i - 1];
	}
	double *shapes_buffer ;
	double *local_buffer = new double[local_size*shape_elements];
	if (process_rank == 0)
	{
		shapes_buffer = new double[shape_size * shape_elements];
		for (int i = 0; i < shape_size; i++)
		{
			for (int j = 0; j < 5; j++)
			{
				shapes_buffer[i*shape_elements + 2 * j]=shapes[i][j].x ;
				shapes_buffer[i*shape_elements + 2 * j + 1]=shapes[i][j].y ;
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	
	// Scatter the shapes data to all processes
	MPI_Scatterv(shapes_buffer, sendcounts, displ,
		MPI_DOUBLE, local_buffer, sendcounts[process_rank], MPI_DOUBLE, 0,MPI_COMM_WORLD);

	vector<vector<cv::Point2d>> local_shapes(local_size, vector<cv::Point2d>(5));

	//fill data into local shapes data
	for (int i = 0; i < local_size; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			local_shapes[i][j].x = local_buffer[i*shape_elements + 2 * j];
			local_shapes[i][j].y = local_buffer[i*shape_elements + 2 * j+1];
		}
	}
    
    // ensure each rank has the same starting mean shape based on landmark count
	double landmarks[LANDMARKS * 2];
	if (process_rank == 0)
	{
		for (int i = 0; i < LANDMARKS; i++)\
		{
			landmarks[2*i]=shapes[0][i].x;
			landmarks[2 * i+1] = shapes[0][i].y;
		}
	}
	MPI_Bcast(landmarks, shape_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Compute the final mean shape using the helper function
	vector<cv::Point2d> final_shape=MeanShape_MPIhelper(local_shapes, landmarks, shape_size);
	
	
	// Uncomment the following block to save the mean shape to a file
    /*
    {
        string filename = "main_train_meanshape_MPIrank"+to_string(process_rank)+".yml";
        cv::FileStorage file;
        file.open(filename, cv::FileStorage::WRITE);
        file << "meanshape" << final_shape;
        file.release();
    }   
    */
    
    // Record the end time and log the time taken for the computation
    const double end_time = MPI_Wtime(); 
    logtime_for_MPI(end_time-begin_time, "MeanShape_MPI");
    
    
	return final_shape;


}


// calculate overall meanshape assuming that all ranks have full data
std::vector<cv::Point2d> MeanShape_MPI(std::vector<std::vector<cv::Point2d>> shapes, const TrainingParameters &tp) {

	// Record the start time
	const double begin_time = MPI_Wtime();

	//vector<vector<cv::Point2d>> shapes;
	int shape_size;
	shape_size = shapes.size();

	//compute the size of each rank
	int local_size = shape_size;



	// ensure each rank has the same starting mean shape based on landmark count
	double landmarks[LANDMARKS * 2];
	if (process_rank == 0)
	{
		for (int i = 0; i < LANDMARKS; i++)\
		{
			landmarks[2 * i] = shapes[0][i].x;
			landmarks[2 * i + 1] = shapes[0][i].y;
		}
	}
	MPI_Bcast(landmarks, LANDMARKS*2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Allreduce(&local_size, &shape_size, 1,
		MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	printf("shape size %ld\n", shape_size);
	// Compute the final mean shape using the helper function
	vector<cv::Point2d> final_shape = MeanShape_MPIhelper(shapes, landmarks, shape_size);

	// Record the end time and log the time taken for the computation
	const double end_time = MPI_Wtime();
	logtime_for_MPI(end_time - begin_time, "MeanShape_MPI");


	return final_shape;
}





