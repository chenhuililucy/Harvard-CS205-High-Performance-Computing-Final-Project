#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include<mpi.h>
#include "mpi_global_vars.h"
#include <omp.h>

void logtime_for_MPI(const double elapsed_time_in_sec, const std::string function_name);