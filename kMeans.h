#pragma once
#include <assert.h>
#include <stdlib.h>
#include <string>

#define ERR(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)


#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);	                                \
    for (size_t i = 0; i < xDim; i++)                       \
        name[i] = (type*)calloc(yDim, sizeof(type));        \
} while (0)

#define matrixCopy(in, out, rows, cols) do {				\
		for(int i = 0; i < rows; i++)					\
		{												\
			for(int j = 0; j < cols; j++)				\
			{											\
				out[i][j] = in[i][j];					\
			}											\
		}												\
	}while(0)
#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
	if (e != cudaSuccess) {
		ERR("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
	}
}

inline void checkLastCudaError() {
	checkCuda(cudaGetLastError());
}
#endif

float** kMeansClustering(float**, int, int, int, float, int*, int*);
float** omp_kmeans(float**, int, int, int, float, int*);
float** seq_kmeans(float**, int, int, int, float, int*, int*);


float** file_read(char*, int*, int* );
void file_write(char*, int, int, int, float**, int*);

float** generatePoints(int numPoints, int numDimensions, float rangeMin = -1500.0, float rangeMax = 1500.0);
void freePoints(float** points, int numDimension);
