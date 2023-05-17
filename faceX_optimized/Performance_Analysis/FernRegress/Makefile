CFLAGS = -std=c++11 -O3 -ggdb -I$(OPENCV_INCLUDE) -fopenmp 
LDFLAGS = -pg -msse2 -march=x86-64 -mavx2
objects = main_train.o fern_train.o regressor_train.o utils_train.o CovarianceISPC.o Covariance.o main_train_MPIfunctions.o logtime_for_MPI.o meanshapeMPI.o


facex-train: $(objects)
		mpic++ $(CFLAGS) $(LDFLAGS) -o facex-train $(objects) -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_video -lopencv_videoio

main_train.o: main_train.cpp regressor_train.h fern_train.h utils_train.h
		mpic++ $(CFLAGS) $(LDFLAGS) -c main_train.cpp -L$(OPENCV_LIB)

fern_train.o: fern_train.cpp fern_train.h utils_train.h
		mpic++ $(CFLAGS) $(LDFLAGS) -c fern_train.cpp -L$(OPENCV_LIB)

regressor_train.o: regressor_train.cpp regressor_train.h fern_train.h utils_train.h
		mpic++ $(CFLAGS) $(LDFLAGS) -c regressor_train.cpp -L$(OPENCV_LIB)

utils_train.o: utils_train.cpp utils_train.h
		mpic++ $(CFLAGS) $(LDFLAGS) -c utils_train.cpp -L$(OPENCV_LIB)

Covariance.o: Covariance.cpp
		mpic++ $(CFLAGS) $(LDFLAGS) -c Covariance.cpp -L$(OPENCV_LIB)
		
CovarianceISPC.o: CovarianceISPC.ispc
		ispc --arch=x86-64 --target=avx2-i32x8 -o $@ $<
		
main_train_MPIfunctions.o: main_train_MPIfunctions.cpp
		mpic++ $(CFLAGS) $(LDFLAGS) -c main_train_MPIfunctions.cpp -L$(OPENCV_LIB)		

logtime_for_MPI.o: logtime_for_MPI.cpp
		mpic++ $(CFLAGS) $(LDFLAGS) -c logtime_for_MPI.cpp -L$(OPENCV_LIB) 
		
meanshapeMPI.o: meanshapeMPI.cpp
		mpic++ $(CFLAGS) $(LDFLAGS) -c meanshapeMPI.cpp -L$(OPENCV_LIB)	
		
clean:
		rm -f main facex-train $(objects) 