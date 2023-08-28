## Brief Introduction

For this project we chose to optimize a microsoft research paper [Cao X, Wei Y, Wen F, et al. Face alignment by explicit shape regression[J]. International Journal of Computer Vision, 2014, 107(2): 177-190.](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/01/Face-Alignment-by-Explicit-Shape-Regression.pdf) with the original code base [here](https://github.com/delphifirst/FaceX)


`faceX` is a C++ program that trains on labeled images and executes facial landmark recognition.

Team 06 of Harvard CS205 class have optimized the training part of `faceX` i.e. `faceX-Train` as their final project in 2023 Spring. 

**Group Members:**

Zhecheng Yao 

Lucy Li 

Rebecca Qiu 

Yixian Gan 

## Usage Guide

Part 1. Sequential FaceX-Train

1. Move to the [faceX_sequential/FaceX-Train](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_sequential/FaceX-Train) directory<br/>
`cd team06/faceX_sequential/FaceX-Train` 

2. Load required modules: <br/>
```
module load git/2.17.0-fasrc01 
module load python/3.9.12-fasrc01 
module load gcc/12.1.0-fasrc01 
module load openmpi/4.1.3-fasrc01 
module load pencv/3.4.3-fasrc01 
module load entos6/0.0.1-fasrc01 
module load pango/1.28.4-fasrc01 
module load eigen/3.3.7-fasrc01 
module load OpenBLAS/0.3.7-fasrc01
```

3. Download training images. The training images can be downloaded from this link: https://drive.google.com/drive/folders/1zFwh5mWGG9ObELvBV4-CrKTxCNur6u2E?usp=share_link. Then upload the images into the [faceX_sequential/FaceX-Train/train/lfw_5590](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_sequential/FaceX-Train/train/lfw_5590) directory. An example image is already provided.

4. Move back to the [faceX_sequential/FaceX-Train](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_sequential/FaceX-Train) directory.

5. [Makefile](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_sequential/FaceX-Train/Makefile) <br/>
`make`

6. Start the training. The training parameters are defined in the [sample_config2.txt](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_sequential/FaceX-Train/sample_config2.txt) file. The training size is defined in the [labels.txt](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_sequential/FaceX-Train/train/labels.txt). Please note that we set the number of regressor to 1 to ensure a reasonable training time, another configuration file that set the number of regressors to 10 is also included in the directory([sample_config.txt](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_sequential/FaceX-Train/sample_config.txt)). Due to this reason, the accuracy of the model may be slightly compromised in exchange for efficiency and you may not get results as accurate as shown in the example output image [result.jpg](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_sequential/FaceX/result.jpg). <br/>
`./facex-train sample_config2.txt`

7. Once the training is finished, the total training time will be printed. The trained model output file ending in .xml.gz ([example](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_sequential/FaceX-Train/model.xml.gz) will show up in the current directory, and it will be used for real-time face landmark detection. Please move or copy it to the [faceX_sequential/FaceX](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_sequential/FaceX) directory. And then move to the  [faceX_sequential/FaceX](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_sequential/FaceX) directory. <br/>
```
cp ./model.xml.gz ../FaceX/model.xml.gz
cd ../FaceX
```

8. [Makefile](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_sequential/FaceX/Makefile) <br/>
`make`

9. Put your test image in the [FaceX directory](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_sequential/FaceX). Rename your test image as [test.jpg](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_sequential/FaceX/test.jpg). An example test image [test.jpg](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_sequential/FaceX/test.jpg) is already provided. Perform real-time testing by entering the command <br/>
`./facex`
 
10. The output image would be named as [result.jpg](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_sequential/FaceX/result.jpg). It will be located in the same [FaceX directory](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_sequential/FaceX).
 

---

Part 2. Optimized FaceX-Train 

1. Request the resource to enable the distributed-memory structure of the optimized function. <br/>
`salloc -N16 -t 2:00:00 --mem=64G`

2. Move to the [faceX_optimized/FaceX-Train](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_optimized/FaceX-Train) directory <br/>
`cd team06/faceX_optimized/FaceX-Train`

3. Similar to the sequential model, user need to download training images. The same training images can be downloaded from this link: https://drive.google.com/drive/folders/1zFwh5mWGG9ObELvBV4-CrKTxCNur6u2E?usp=share_link. Then upload the images into the [faceX_optimized/FaceX-Train/train_large/lfw_5590](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_optimized/FaceX-Train/train_large/lfw_5590) directory. An example image is already provided.

4. Move back to the [faceX_optimized/FaceX-Train](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_optimized/FaceX-Train) directory and start the training. We provide a job script [run_time.sh](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX-Train/run_time.sh) to conveniently train the model using different number of images across different number of nodes. The number of images we choose here are 250, 500, 1000, 2000, and 4000. The number of nodes we choose here are from 1 to 16. <br/>
`sbatch run_time.sh` <br/>
Please note that this step performs strong scaling analysis for each MPI kernel of the optimized FaceX-Train while training the model to determine the optimal number of nodes, which will take a relatively long time to complete the whole process. If the user wish to skip the scaling analysis and directly train the model, run the following commands instead and skip to step 11.<br/>
```
module load git/2.17.0-fasrc01 
module load python/3.9.12-fasrc01 
module load gcc/12.1.0-fasrc01 
module load openmpi/4.1.3-fasrc01 
module load pencv/3.4.3-fasrc01 
module load entos6/0.0.1-fasrc01 
module load pango/1.28.4-fasrc01 
module load eigen/3.3.7-fasrc01 
module load OpenBLAS/0.3.7-fasrc01

```
`make` <br/>
`srun --ntasks 5 ./facex-train config.txt` 

5. Within the run_time.sh script, at each step, the python script [generate_labels.py](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX-Train/generate_labels.py) creates a random set of images with the defined training size. 

6. Then the same set of images is used to trained on different number of nodes with the same training configuration defined in the [config.txt](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX-Train/config.txt)

7. During the training, the wall time of each MPI-based kernel are recorded in the faceX_optimized/FaceX-Train/[logs](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_optimized/FaceX-Train/logs) directory using the [logtime_for_MPI.cpp](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX-Train/logtime_for_MPI.cpp) C++ script.

8. Once the training is done, the wall times for each MPI kernel with the same training size are combined into one log file and saved in the faceX_optimized/FaceX-Train/[combined_logs](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_optimized/FaceX-Train/combined_logs) directory using the [plot.py](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX-Train/plot.py) python script. 

9. The plot.py script accomplishes multiple things. 
* It first plots the measured wall time against the number of nodes for each 
MPI-based kernel from the combined_log files (see example plots for the [GetTrainingData_MPI()](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX-Train/plots/Time%20Measurement%20for%20Optimized%20GetTrainingData_MPI%20Training%20over%20Different%20Number%20of%20Images.png) kernel. 

* It then plots the strong scaling graph for each MPI-based kernel (see example plot for the [GetTrainingData_MPI()](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX-Train/plots/Strong%20Scaling%20Plot%20for%20Optimized%20GetTrainingData_MPI%20Training%20over%20Different%20Number%20of%20Images.png). 

* All the plots are saved in the faceX_optimized/FaceX-Train/[plots](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_optimized/FaceX-Train/plots) directory. 

10. Please note that steps 5 - 9 are detailed explanations of what the [run_time.sh](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX-Train/run_time.sh) accomplishes internally, no actions need to be taken on the user's end. The whole training process will take a while because the model is trained repeatedly from using 1 rank to 16 ranks to determine the optimal number of nodes.

11. Once the training is finished, move or copy the trained model output file ending in .xml.gz ([example](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX-Train/model.xml.gz)) to the [faceX_optimized/FaceX](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_optimized/FaceX) directory. <br/> Then move to the [faceX_optimized/FaceX](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_optimized/FaceX) directory.
```
cp ./model.xml.gz ../FaceX/model.xml.gz
cd ../FaceX
```

12. [Makefile](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX/Makefile) <br/>
`make`

13. Put your test image in the [FaceX directory](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX). Rename your test image as [test.jpg](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX/test.jpg). Perform real-time testing by entering the command <br/>
`./facex`
 
14. The output image would be named as [result.jpg](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX/result.jpg) It will be located in the same [FaceX directory](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX).


---


Part 3. Procrustes Kernel

1. Move to the faceX_optimized/Performance_Analysis/[Procrustes](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_optimized/Performance_Analysis/Procrustes) directory <br/>
`cd team06/faceX_optimized/Performance_Analysis/Procrustes`

2. Reproduce the test results by running the [test_Procrustes.sh](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/Performance_Analysis/Procrustes/test_Procrustes.sh) job script. <br/>
`sbatch test_Procrustes.sh`

3. Inspect the output file test.out to see the performance of both the sequential and the optimized functions. You will most likely see something similar to the [example output](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/Performance_Analysis/Procrustes/test_example.out) below:<br/>

```
Average execution time of Procrustes: 41.7 ms
Average execution time of ProcrustesV: 25 ms
Procrustes and ProcrustesV produce the same result

Data Size: 20000000 2dPoint
Average execution time of Procrustes: 82.4 ms
Average execution time of ProcrustesV: 50.2 ms
Procrustes and ProcrustesV produce the same result

Data Size: 30000000 2dPoint
Average execution time of Procrustes: 125 ms
Average execution time of ProcrustesV: 76 ms
Procrustes and ProcrustesV produce the same result

Data Size: 40000000 2dPoint
Average execution time of Procrustes: 166.2 ms
Average execution time of ProcrustesV: 101.9 ms
Procrustes and ProcrustesV produce the same result

[roofline analysis](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/Performance_Analysis/Procrustes/procrustes_roofline.png) 
[scaling analysis](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/Performance_Analysis/Procrustes/procrustes_scaling.png) 
```
The job scripts calls [`Procrustes.cpp`](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/Performance_Analysis/Procrustes/Procrustes.cpp), which randomly generates one set of 2dPoint objects and feed into both the original Procrustes() function and the vectorized ProcrustesV() function. The test performs 2 tasks: measure the wall time of each kernel and compare the computated transformation vectors produced by each. If the results are identical, then the ProcrustesV() passes the test. The measured wall times are used to calculate the operational intensity of both kernels as well. <br/>

The same procedure is repeated 3 more times with increasing data size to see how the performance of both kernels scale to problem size.<br/>

The links to the plots of [roofline analysis](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/Performance_Analysis/Procrustes/procrustes_roofline.png) and [scaling analysis](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/Performance_Analysis/Procrustes/procrustes_scaling.png) of the example test case are also provided in the output file. <br/>



---



Part 4. [Orthogonal Matching Pursuit Kernel](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/Performance_Analysis/Orthogonal_Matching_Pursuit/utils_train.cpp#L316)

1. Move to the faceX_optimized/Performance_Analysis/[Orthogonal_Matching_Pursuit](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_optimized/Performance_Analysis/Orthogonal_Matching_Pursuit) directory <br/>
 `cd team06/faceX_optimized/Performance_Analysis/Orthogonal_Matching_Pursuit`

 2. Place the downloaded dataset from this link: https://drive.google.com/drive/folders/1zFwh5mWGG9ObELvBV4-CrKTxCNur6u2E?usp=share_link. Then upload the images into the [FaceX_optimized/Performance_Analysis/Orthogonal_Matching_Pursuit/train2/lfw_5590]

 4. Reproduce the test results by running the [test_OMP.sh](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/Performance_Analysis/Orthogonal_Matching_Pursuit/test_OMP.sh)
 job script. <br/> `sbatch test_OMP.sh`


 3. Inspect the output file, the same step is repeated for different values of Q: 5, 20, 30, 40, 50.


 ---
 ```
======= New Iteration Starts ========
The parameter Q for this iteration is:
5

Performance Analysis Eigen library
Total cycles:                 1232480
Total instructions:           1980421
Instructions per cycle (IPC): 1.60686
L1 cache size:                32 KB
L2 cache size:                256 KB
L3 cache size:                40960 KB
Total L1 data misses:         6933
Total load/store:             812696
 (expected: 36031)
Operational intensity:        8.420584e-03
 (expected: 1.899302e-01)
Performance [Gflop/s]:        1.054514e-01
Wall-time   [micro-seconds]:  5.191680e+02

Performance Analysis original Library 
Total cycles:                 8620322
Total instructions:           14454190
Instructions per cycle (IPC): 1.676758e+00
L1 cache size:                32 KB
L2 cache size:                256 KB
L3 cache size:                40960 KB
Total L1 data misses:         27830
Total load/store:             8271750
Operational intensity:        8.273189e-04
Performance [Gflop/s]:        1.834095e-02
Wall-time   [micro-seconds]:  2.984960e+03

 ```


---

Part 5. Covariance Kernel
1. Move to this directory [faceX_optimized/Performance_Analysis/Covariance](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_optimized/Performance_Analysis/Covariance) <br/>
`cd team06/faceX_optimized/Performance_Analysis/Covariance`

2. Reproduce roofline analysis and time measurement data as well as programmatically plotting their graphs based on such data:<br/>
`sbatch run.sh `

The [`run.sh`](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/Performance_Analysis/Covariance/run.sh) script calls [`main.cpp`](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/Performance_Analysis/Covariance/main.cpp) which randomly generates 2 arrays of doubles and feed into both the original Covariance() function and the ISPC vectorized Covariance() function. `main.cpp` first checks if the results calculated from each variation of Covariance() are within 0.00001 absolute difference. If so it is considered result matched and will continue to run each function 10000 times. `main.cpp` program will measuring the wall time of each kernel and output the average runtime. 

Then `run.sh` script will call the [`plot.py`](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/Performance_Analysis/Covariance/plot.py) program which will read in the average runtimes and calculate the operational intensity and performance. Finally `plot.py` program calculate and plot roofline analysis and time measurements following the logic described in our report section 4.

Use the time information and following the math logic described in this [document](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/Performance_Analysis/Covariance/Performance%20and%20Operational%20Intensity%20Calculation%20for%20Covariance%20Kernel.pdf), you should be able to plot the roofline analysis and time measurement graphs.

You can find the output data located under `team06/faceX_optimized/Performance_Analysis/Covariance/.` folder called "wall_time_results for Covariance Kernel.log" similar to this one in Git repo example: [wall_time_results for Covariance Kernel.log](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/Performance_Analysis/Covariance/wall_time_results%20for%20Covariance%20Kernel.log)

The sample plot graphs for 16000 data size can be found in the same directory. The corresponding time measurement used for those plots are based on the above example log file. The plot names are [Time Measurement of Covariance Kernel.png](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/Performance_Analysis/Covariance/Time%20Measurement%20of%20Covariance%20Kernel.png) and [Roofline analysis for Covariance Kernel.png](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/Performance_Analysis/Covariance/Roofline%20analysis%20for%20Covariance%20Kernel.png)


---

Part 6. Fern Regressor Kernel

1. Direct to the [faceX_optimized/Performance_Analysis/FernRegress](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_optimized/Performance_Analysis/FernRegress) directory <br/>
`cd team06/faceX_optimized/Performance_Analysis/FernRegress`

2. Copy your training images folder to [`train/`](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_optimized/Performance_Analysis/FernRegress/train) directory. Currently the `/train` directory provides a [`labels.txt`](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/Performance_Analysis/FernRegress/train/labels.txt) that contains labels for training images in the `lfw_5590` dataset. If you want to use other dataset, please replace this label file with your label data. *Make sure your label file is also named `labels.txt`*

3. Inside `faceX_optimized/Performance_Analysis/FernRegress/`, submit the job script by <\br>
`sbatch run.sh`

4. You can find the [runtime](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/Performance_Analysis/FernRegress/plots/Time%20Measurement%20for%20Optimized%20Fern_train%20Training%20over%20Different%20Number%20of%20Images.png) and [strong scaling](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/Performance_Analysis/FernRegress/plots/Strong%20Scaling%20Plot%20for%20Optimized%20Fern_train%20Training%20over%20Different%20Number%20of%20Images.png) plots of `Fern::Regress` function under the [`plots/`](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_optimized/Performance_Analysis/FernRegress/plots) directory. The plot would also show that how is the runtime and strong scaling result affected as size of training data changes (each plot contains five curves representing 250, 500, 1000, 2000, and 4000 training images used). You can also find several `*.log` files in the same directory. Those are raw data of running time. For example, [`wall_time_results for Fern_train using 1000 images.log`](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/Performance_Analysis/FernRegress/plots/wall_time_results%20for%20Fern_train%20using%201000%20images.log) records the execution time of training one Fern regressor on 1000 training images.

---

Part 7. Overall

To perform strong scaling analysis and determine the optimal number of nodes for the entire training process, follow the instructions below. Please note that is section is different from Part 2 above in the sense that the steps below are used to measure the overall runtime of the optimized Facex-Train instead of each individual MPI kernel. The  [logtime_for_MPI.cpp](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX-Train/logtime_for_MPI.cpp) script is called for computing the total training time only. 

1. Request the resource to enable the distributed-memory structure of the optimized function. <br/>
`salloc -N16 -t 2:00:00 --mem=64G`

2. Move to the [faceX_optimized/FaceX-Train](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_optimized/FaceX-Train) directory <br/>
`cd team06/faceX_optimized/FaceX-Train`

3. Download training images.The training images can be downloaded from this link: https://drive.google.com/drive/folders/1zFwh5mWGG9ObELvBV4-CrKTxCNur6u2E?usp=share_link. Then upload the images into the [faceX_optimized/FaceX-Train/train_large](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_optimized/FaceX-Train/train_large/lfw_5590) directory. An example image is already provided.

4. Move back to the [faceX_optimized/FaceX-Train](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_optimized/FaceX-Train) directory and start the training. We provide a job script [run_time.sh](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX-Train/run_time.sh) to conveniently train the model using different number of images across different number of nodes. The number of images we choose here are 400, 800, 1000, 2000, and 4000. The number of nodes we choose here are from 1 to 8. <br/>
`sbatch run_time.sh`

5. Within the run_time.sh script, at each step, the python script [generate_labels.py](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX-Train/generate_labels.py) creates a random set of images with the defined training size. 

6. Then the same set of images is used to trained on different number of nodes with the same training configuration defined in the [train_large_config.txt](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX-Train/train_large_config.txt).

7. During the training, the wall time of the overall training is recorded in the [faceX_optimized/FaceX-Train/logs](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_optimized/FaceX-Train/logs) directory using the [logtime_for_MPI.cpp](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX-Train/logtime_for_MPI.cpp) C++ script.

8. Once the training is done, the wall times with the same training size are combined into one log file and saved in the [faceX_optimized/FaceX-Train/combined_logs/](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_optimized/FaceX-Train/combined_logs) directory using the [plot.py](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX-Train/plot.py) python script. 

9. The plot.py script accomplishes multiple things. 
* It first plots the measured wall time against the number of nodes for the overall training from the combined_log files (see example plots for the overall optimized [FaceX-Train](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX-Train/plots/Time%20Measurement%20for%20Optimized%20FaceX-Train%20Training%20over%20Different%20Number%20of%20Images.png)) model. 

* It then plots the strong scaling graph for the overall training (see example plot for the overall [FaceX-Train](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX-Train/plots/Strong%20Scaling%20Plot%20for%20Optimized%20FaceX-Train%20Training%20over%20Different%20Number%20of%20Images.png)). 

* Finally, based on the strong scaling analysis, it determines the optimal number of nodes for each training size and creates the corresponding plots (see example plot for the optimized [FaceX-Train](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX-Train/plots/Optimal%20Node%20for%20Optimized%20FaceX-Train%20Training%20over%20Different%20Number%20of%20Images.png)). 

* All the plots are saved in the [FaceX_optimized/FaceX-Train/plots](https://github.com/chenhuililucy/CS205_final_project/tree/main/faceX_optimized/FaceX-Train/plots) directory. 

10. Please note that steps 5 - 9 are detailed explanations of what the [run_time.sh](https://github.com/chenhuililucy/CS205_final_project/blob/main/faceX_optimized/FaceX-Train/run_time.sh) accomplishes internally, no actions need to be taken on the user's end. The whole training process will take a while because the model is trained repeatedly from using 1 rank to 8 ranks to determine the optimal number of nodes.
