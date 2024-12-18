#include "kMeans.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>

static inline int nextPowerOfTwo(int n) {
	n--;

	n = n >> 1 | n;
	n = n >> 2 | n;
	n = n >> 4 | n;
	n = n >> 8 | n;
	n = n >> 16 | n;

	return ++n;
}

__host__ __device__ inline static
float euclid_dist_2(int numDimension,
	int numCluster,
	int numPoints,
	float* data,
	float* clusters,
	int pointId,
	int clusterId)
{
	float ans = 0;
	for (int i = 0; i < numDimension; i++)
	{
		ans += (data[i * numPoints + pointId] - clusters[i * numCluster + clusterId]) * (data[i * numPoints + pointId] - clusters[i * numCluster + clusterId]);
	}
	return ans;
}

__global__ static
void find_nearest_cluster(int numDimension,
	int numPoint,
	int numCluster,
	float* points,
	float* deviceClusters,
	int* membership,
	int* intermediates)
{
	extern __shared__ char sharedMemory[];
	unsigned char* membership_changed = (unsigned char*)sharedMemory;
	float* clusters = deviceClusters;

	membership_changed[threadIdx.x] = 0;

	int pointId = blockDim.x * blockIdx.x + threadIdx.x;

	if (pointId < numPoint)
	{
		float dist, minDist;
		
		int clusterIdx = 0;
		minDist = euclid_dist_2(numDimension, numCluster, numPoint ,points, deviceClusters, pointId, 0);
		for (int i = 1; i < numCluster; i++)
		{
			dist = euclid_dist_2(numDimension, numCluster, numPoint, points, deviceClusters, pointId, i);
			if (dist < minDist)
			{
				clusterIdx = i;
				minDist = dist;
			}
		}

		if (membership[threadIdx.x] != clusterIdx)
		{
			membership_changed[threadIdx.x] = 1;
		}

		membership[pointId] = clusterIdx;

		__syncthreads();

		//  blockDim.x *must* be a power of two!
		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (threadIdx.x < s) {
				membership_changed[threadIdx.x] +=
					membership_changed[threadIdx.x + s];
			}
			__syncthreads();
		}

		if (threadIdx.x == 0) {
			intermediates[blockIdx.x] = membership_changed[0];
		}
	}
}

__global__ static
void compute_delta(int* deviceIntermediates,
	int numIntermediates,
	int numIntermediates2)
{
	extern __shared__ unsigned int intermediates[];

	//  Copy global intermediate values into shared memory.
	intermediates[threadIdx.x] = (threadIdx.x < numIntermediates) ? deviceIntermediates[threadIdx.x] : 0;

	__syncthreads();

	// reduction
	for (unsigned int s = numIntermediates2 / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			intermediates[threadIdx.x] += intermediates[threadIdx.x + s];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		deviceIntermediates[0] = intermediates[0];
	}
}


/*
* input: points - coordinates of points
*		numPoints - number of points
*		numDimensions - number of dimensions
*		numClusters - number of clusters
*		threshold - % of points to change their membership
* output: membership - allocation of points to centroids
*		iterations - number of iterations to compute centroids
*/
float** kMeansClustering(float** points,
						int numPoints,
						int numDimensions,
						int numClusters,
						float threshold,
						int* membership,
						int* iterations)
{
	int      i, j, index, loop = 0;
	int* newClusterSize;
	float    delta;          /* % of points change their clusters */
	float** dimPoints;
	float** clusters;
	float** dimClusters;
	float** newClusters;

	float* devicePoints;
	float* deviceClusters;
	int* deviceMembership;
	int* deviceIntermediates;

	malloc2D(dimPoints, numDimensions, numPoints, float);
	matrixCopy(points, dimPoints, numDimensions, numPoints);

	malloc2D(dimClusters, numDimensions, numClusters, float);
	matrixCopy(dimPoints, dimClusters, numDimensions, numClusters);

	for (int i = 0; i < numPoints; i++)
	{
		membership[i] = -1;
	}

	newClusterSize = (int*)calloc(numClusters, sizeof(int));
	assert(newClusterSize != NULL);

	malloc2D(newClusters, numDimensions, numClusters, float);
	memset(newClusters[0], 0, numClusters * numDimensions * sizeof(float));

	const unsigned int numThreadsPerClusterBlock = 128;
	const unsigned int numClusterBlocks = (numPoints + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;
	// No shared_memory optimization
	const unsigned int clusterBlockSharedDataSize = numThreadsPerClusterBlock * sizeof(unsigned char);

	const unsigned int numReductionThreads = nextPowerOfTwo(numClusterBlocks);
	const unsigned int reductionBlockSharedDataSize = numReductionThreads * sizeof(unsigned int);

	cudaMalloc(&devicePoints, numPoints * numDimensions * sizeof(float));
	cudaMalloc(&deviceClusters, numClusters * numDimensions * sizeof(float));
	cudaMalloc(&deviceMembership, numPoints * sizeof(int));
	cudaMalloc(&deviceIntermediates, numReductionThreads * sizeof(unsigned int));

	cudaMemcpy(devicePoints, dimPoints[0], numPoints * numDimensions * sizeof(float), cudaMemcpyHostToDevice);
	checkCuda(cudaMemcpy(deviceMembership, membership, numPoints * sizeof(int), cudaMemcpyHostToDevice));

	do {
		checkCuda(cudaMemcpy(deviceClusters, dimClusters[0], numClusters * numDimensions * sizeof(float), cudaMemcpyHostToDevice));

		find_nearest_cluster <<<numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>> (numDimensions, numPoints, numClusters, devicePoints, deviceClusters, deviceMembership, deviceIntermediates);

		cudaDeviceSynchronize();
		checkLastCudaError();

		compute_delta <<< 1, numReductionThreads, reductionBlockSharedDataSize >> >
			(deviceIntermediates, numClusterBlocks, numReductionThreads);

		cudaDeviceSynchronize(); 
		checkLastCudaError();

		int d;
		checkCuda(cudaMemcpy(&d, deviceIntermediates, sizeof(int), cudaMemcpyDeviceToHost));
		delta = (float)d;

		checkCuda(cudaMemcpy(membership, deviceMembership, numPoints * sizeof(int), cudaMemcpyDeviceToHost));

		for (i = 0; i < numPoints; i++) {
			index = membership[i];

			newClusterSize[index]++;
			for (j = 0; j < numDimensions; j++)
			{
				newClusters[j][index] += points[j][i];
			}
		}
		for (i = 0; i < numClusters; i++) {
			for (j = 0; j < numDimensions; j++) {
				if (newClusterSize[i] > 0)
				{
					dimClusters[j][i] = newClusters[j][i] / newClusterSize[i];
				}
				newClusters[j][i] = 0.0;
			}
			newClusterSize[i] = 0;
		}

		delta /= numPoints;
	} while (delta > threshold && loop++ < 500);

	*iterations = loop + 1;

	malloc2D(clusters, numDimensions, numClusters, float);
	matrixCopy(dimClusters, clusters, numDimensions, numClusters);

	checkCuda(cudaFree(devicePoints));
	checkCuda(cudaFree(deviceClusters));
	checkCuda(cudaFree(deviceMembership));
	checkCuda(cudaFree(deviceIntermediates));

	free(dimPoints[0]);
	free(dimPoints);
	free(dimClusters[0]);
	free(dimClusters);
	//free(newClusters[0]);
	free(newClusters);
	free(newClusterSize);

	return clusters;
}