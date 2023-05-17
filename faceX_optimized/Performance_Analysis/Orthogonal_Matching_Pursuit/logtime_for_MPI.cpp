#include "logtime_for_MPI.h"

using namespace std;

void logtime_for_MPI(const double elapsed_time_in_sec, const string function_name )
{
    double globalWallTimeSum, globalWallTimeSqDiff, globalWallTimeStdDev;

    MPI_Allreduce(&elapsed_time_in_sec, &globalWallTimeSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double globalWallTimeMean = globalWallTimeSum/process_cnt;
    const double localDeviationSq =  (elapsed_time_in_sec - globalWallTimeMean)*(elapsed_time_in_sec - globalWallTimeMean);
    MPI_Allreduce(&localDeviationSq, &globalWallTimeSqDiff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    globalWallTimeStdDev = sqrt(globalWallTimeSqDiff/process_cnt);
    
    if (process_rank == 0) {
        const string filename = "wall_time_results for "+function_name +".log";
        ifstream inFile(filename);
        vector<string> fileLines;
        // Read existing lines from the file
        if (inFile) {
            string line;
            while (getline(inFile, line)) {
                fileLines.push_back(line);
            }
            inFile.close();
        }

        // Update the lines with the new results
        if (fileLines.size() < 1) {
            fileLines.push_back("Process Count; Mean Wall Time(in sec); Standard Deviation(in sec)");
            fileLines.push_back(function_name);
            fileLines.push_back(to_string(process_cnt));
            fileLines.push_back(to_string(globalWallTimeMean));
            fileLines.push_back(to_string(globalWallTimeStdDev));
        } else{
            fileLines[2] += ", " + to_string(process_cnt);
            fileLines[3] += ", " + to_string(globalWallTimeMean);
            fileLines[4] += ", " + to_string(globalWallTimeStdDev);
        }
        // Rewrite the file with the updated content
        ofstream outFile(filename);
        for (const auto& line : fileLines) {
            outFile << line << endl;
        }
        outFile.close();
    }    
}