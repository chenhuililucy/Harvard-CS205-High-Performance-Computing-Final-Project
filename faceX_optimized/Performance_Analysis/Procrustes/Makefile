CC=g++
CFLAGS= -std=c++17 -O3 -mavx -fopenmp -lpapi -march=native
LDFLAGS=-lopencv_core -lopencv_imgproc -lopencv_highgui

all: Procrustes

Procrustes: Procrustes.cpp
	$(CC) $(CFLAGS) Procrustes.cpp -o Procrustes $(LDFLAGS)

clean:
	rm -f Procrustes