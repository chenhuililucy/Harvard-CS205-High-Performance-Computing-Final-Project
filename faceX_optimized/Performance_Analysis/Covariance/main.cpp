
#include "Covariance.h"
#include "testlib.h"


using namespace std;



int main(int argc, char** argv) {

    if (argc != 2) {
        cout << "Usage: ./program_name array_size" << endl;
        return 1;
    }
    int array_size = stoi(argv[1]);

    // define test array size based on a real image sample
    //int array_size = 2693;
    cout<< "array size is " << array_size << endl;
    double* a = new double[array_size];
    double* b = new double[array_size];
    // align to 256 bit vector size
    posix_memalign((void **)&a, 32, array_size * sizeof(double));
    posix_memalign((void **)&b, 32, array_size * sizeof(double));

    // we use typical input value for x and y provided by faceX models
    double lower_bound_a = -10;
    double upper_bound_a = 10;
    double lower_bound_b = 1;
    double upper_bound_b = 500;

    // test if Covariance calculation results are the same
    const double diff_threshold = 1e-5;
    double test1, test2, test3;
    test1 = Covariance(a, b, array_size);
    test2 = CovarianceISPC(a, b, array_size);
    if(abs(test2-test1)<=diff_threshold){
        cout<<"result match"<<endl;
    }
    else{
        return 1;
    }



    // initialize test data arrays with random generators
//#pragma omp parallel for schedule(static)
    for (int i = 0; i < array_size; i++) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis_a(lower_bound_a, upper_bound_a);
        std::uniform_real_distribution<> dis_b(lower_bound_b, upper_bound_b);
        a[i] = dis_a(gen);
        b[i] = dis_b(gen);
    }

    // set parameters for testing
    const int test_loop_cnt = 1e4; // how many experiment rounds to run
    const int test_size = 1; // in each experiment round the covariance function call time
    const int function_size = 1; // the actual function call in a real image sample
    //cout<< *a <<";" << *b<<";"<< c_random_int<<endl;
    double sum1 = 0;

    // test using serial code covariance computation
    for (int test_idx =0; test_idx < test_loop_cnt; ++test_idx){
//#pragma omp parallel for reduction(+:sum1) schedule(static)
        for (int i = 0; i < test_size; ++i){
            auto start1 = omp_get_wtime();
            double test1 = Covariance(a, b, array_size);
            auto end1 = omp_get_wtime();
            double diff1 = end1 - start1;
            //cout<<test1<<endl;
            //std::cout << "Time taken is: " << diff.count() << " s\n";
            sum1+= diff1;
        }
    }
    double avg_time1 = (sum1/(test_loop_cnt*test_size))*function_size;
    cout<<"Estimated Total Function Time in secs by normal Covariance: " << avg_time1 << " seconds"<<endl;


    // test using ISPC covariance computation
    double sum2 = 0;
    for (int test_idx =0; test_idx < test_loop_cnt; ++test_idx){
//#pragma omp parallel for reduction(+:sum2) schedule(static)
        for (int i = 0; i < test_size; ++i){
            auto start2 = omp_get_wtime();
            double test2 = CovarianceISPC(a, b, array_size);
            auto end2 = omp_get_wtime();
            double diff2 = end2 - start2;
            //cout<<test2<<endl;
            //std::cout << "Time taken is: " << diff.count() << " s\n";
            sum2+= diff2;
        }
    }
    double avg_time2 = (sum2/(test_loop_cnt*test_size))*function_size;
    cout<<"Estimated Total Function Time in secs by ISPC Covariance: " << avg_time2 << " seconds"<<endl;


    const string directory = "./";
	const string function_name = "Covariance Kernel";
    const string filename = directory +"wall_time_results for "+function_name +".log";
	const int scale_factor = 1e6;
	const int shrink_factor = 1e3;
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
        fileLines.push_back("Data Size(in thousands); Sequential Performance(in microseconds); ISPC Performance(in microseconds)");
        fileLines.push_back(function_name);
		fileLines.push_back(to_string(array_size/shrink_factor));
        fileLines.push_back(to_string(avg_time1*scale_factor));
        fileLines.push_back(to_string(avg_time2*scale_factor));
    } else{
		fileLines[2] += ", " + to_string(array_size/shrink_factor);
        fileLines[3] += ", " + to_string(avg_time1*scale_factor);
        fileLines[4] += ", " + to_string(avg_time2*scale_factor);
    }
    // Rewrite the file with the updated content
    ofstream outFile(filename);
    for (const auto& line : fileLines) {
        outFile << line << endl;
    }
    outFile.close();

    delete[] a;
    delete[] b;
    return 0;
}