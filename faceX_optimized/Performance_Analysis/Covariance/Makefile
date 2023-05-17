CFLAGS = -std=c++11 -O2 -ggdb -fopenmp -march=x86-64 -mavx2 -mfma
LDFLAGS =  -pg 
objects = main.o CovarianceISPC.o Covariance.o

main: $(objects)
		mpic++ $(CFLAGS) $(LDFLAGS) $(objects) -o main

main.o: main.cpp 
		mpic++ $(CFLAGS) -c main.cpp 

CovarianceISPC.o: CovarianceISPC.ispc
		ispc --arch=x86-64 --target=avx2-i64x4 -o $@ $<

Covariance.o: Covariance.cpp
		mpic++ $(CFLAGS) $(LDFLAGS) -c Covariance.cpp

clean:
		rm -f main $(objects)
