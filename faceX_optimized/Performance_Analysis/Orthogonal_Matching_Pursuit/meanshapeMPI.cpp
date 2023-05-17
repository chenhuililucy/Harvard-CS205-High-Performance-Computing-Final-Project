#include "meanshapeMPI.h"





#include "mpi_global_vars.h"
#include <omp.h>


using namespace std;

#define LANDMARKS 5



static void Normalize(vector<cv::Point2d> * shape)
{

	cv::Point2d center;
	for (const cv::Point2d p : *shape)
		center += p;
	center *= 1.0 / shape->size();
	for (cv::Point2d &p : *shape)
		p -= center;
	
	const int left_eye_index = 0;
	const int right_eye_index = 1;

	cv::Point2d left_eye = (*shape).at(left_eye_index);
	cv::Point2d right_eye = (*shape).at(right_eye_index);
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

vector<cv::Point2d> MeanShape_MPIhelper(vector<vector<cv::Point2d>> shapes,double* inital_marks,int total_size)
{
	const int kIterationCount = 10;
	double reduce_shapes[LANDMARKS * 2];
	vector<cv::Point2d> mean_shape (LANDMARKS);
	for (int i = 0; i < LANDMARKS; i++)
	{
		mean_shape[i].x = inital_marks[2 * i];
		mean_shape[i].y = inital_marks[2 * i+1];
	}
	for (int i = 0; i < kIterationCount; ++i)
	{
		for (vector<cv::Point2d> & shape : shapes)
		{
			Transform t = Procrustes(mean_shape, shape);
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
		for (int j = 0; j < LANDMARKS; j++)
		{
			inital_marks[2 * j]=mean_shape[j].x;
			inital_marks[2 * j + 1]= mean_shape[j].y ;
		}
		MPI_Allreduce(inital_marks, reduce_shapes, 10, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		for (int j = 0; j < LANDMARKS; j++)
		{
			mean_shape[j].x= reduce_shapes[2 * j]  ;
			mean_shape[j].y = reduce_shapes[2 * j + 1];
		}
		for (cv::Point2d & p : mean_shape)
			p *= 1.0 / total_size;

		Normalize(&mean_shape);
	}

	return mean_shape;
}

std::vector<cv::Point2d> MeanShape_MPI(std::vector<std::vector<cv::Point2d>> shapes, const TrainingParameters &tp){
    
	//vector<vector<cv::Point2d>> shapes;
	int shape_size;
	shape_size=shapes.size();

	//bcast the shape size to all ranks
	MPI_Bcast(&shape_size, 1, MPI_INT, 0, MPI_COMM_WORLD);


	//compute the size of each rank
	int local_size = shape_size / process_cnt;
	int remain = (shape_size % process_cnt);
	if (process_rank < remain)
		local_size += 1;
	// number of double values for each shape;
	int shape_elements = 10;
	
	//scatter buffer to each rank
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

	
	vector<cv::Point2d> final_shape=MeanShape_MPIhelper(local_shapes, landmarks, shape_size);
    /*
    {
        string filename = "main_train_meanshape_MPIrank"+to_string(process_rank)+".yml";
        cv::FileStorage file;
        file.open(filename, cv::FileStorage::WRITE);
        file << "meanshape" << final_shape;
        file.release();
    }   
    */
	return final_shape;
	//printf to check the result 

/*	//free buffer
	if(process_rank==0){
    	delete[]shapes_buffer;
    	delete[]local_buffer;
    	delete[]sendcounts;
    	delete[]displ;
	}*/
}



