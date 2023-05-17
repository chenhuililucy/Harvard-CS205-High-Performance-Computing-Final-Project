#pragma once


#include "utils_train.h"
#include "mpi.h"
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <map>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>


std::vector<cv::Point2d> MeanShape_MPI(std::vector<std::vector<cv::Point2d>> shapes, const TrainingParameters &tp);
std::vector<cv::Point2d> MeanShape_local_MPI(std::vector<std::vector<cv::Point2d>> shapes, const TrainingParameters &tp);
static void Normalize(std::vector<cv::Point2d> * shape);
std::vector<cv::Point2d> MeanShape_MPIhelper(std::vector<std::vector<cv::Point2d>> shapes,double* inital_marks,int total_size);