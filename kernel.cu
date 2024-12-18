#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kMeans.h"
#include <stdio.h>
#include <iostream>

#define NUM_CLUSTERS 3
#define DELTA 0.01

int main()
{
    int numPoints = 0;
    int dimension = 0;
    int* membership;    /* [numObjs] */
    char* filename;
    float** points;       /* [numObjs][numCoords] data objects */
    float** clusters;      /* [numClusters][numCoords] cluster center */
    float   threshold;
    double  timing, io_timing, clustering_timing;
    int     loop_iterations;

    /* read data points from file ------------------------------------------*/
    points = file_read("text.txt", &numPoints, &dimension);
    assert(points != NULL);

    /* start the timer for the core computation -----------------------------*/
    /* membership: the cluster id for each data object */
    membership = (int*)malloc(numPoints * sizeof(int));
    assert(membership != NULL);

    clusters = kMeansClustering(points, numPoints, dimension, NUM_CLUSTERS, DELTA, membership, &loop_iterations);


    /* output: the coordinates of the cluster centres ----------------------*/
    file_write("result.txt", NUM_CLUSTERS, numPoints, dimension, clusters, membership);

    /* start the timer for the core computation -----------------------------*/
    /* membership: the cluster id for each data object */
    membership = (int*)malloc(numPoints * sizeof(int));
    assert(membership != NULL);

    clusters = omp_kmeans(points, numPoints, dimension, NUM_CLUSTERS, DELTA, membership);

    free(points);

    /* output: the coordinates of the cluster centres ----------------------*/
    file_write("result_omp.txt", NUM_CLUSTERS, numPoints, dimension, clusters, membership);

    free(membership);
    free(clusters[0]);
    free(clusters);
}
