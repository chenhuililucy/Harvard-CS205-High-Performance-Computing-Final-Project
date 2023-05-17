#include "utils_train.h"


void serializeDataPoint(const DataPoint &dp, std::vector<char> &buffer);

DataPoint deserializeDataPoint(const std::vector<char> &buffer, size_t &offset);

std::vector<DataPoint> GetTrainingData_MPI(const TrainingParameters &tp);

std::vector<std::vector<cv::Point2d>> CreateTestInitShapes_MPI( const std::vector<DataPoint>& training_data, const TrainingParameters& tp);

std::vector<DataPoint> ArgumentData_MPI(const std::vector<DataPoint> &training_data, int factor);