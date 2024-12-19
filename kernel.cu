#include "kMeans.h"
#include <stdio.h>
#include <iostream>
#include <chrono>

#define NUM_CLUSTERS 3
#define DELTA 0.01
#define NUM_POINTS 100000
#define NUM_DIMENSION 96

int main()
{
    int numPoints = 0;
    int dimension = 0;
    int* membership;
    char* filename;
    float** points;
    float** clusters;
    float   threshold;
    double  timing, io_timing, clustering_timing;
    int     loop_iterations;
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::microseconds duration;
    

    /* read data points from file ------------------------------------------*/
    points = file_read("text.txt", &numPoints, &dimension);
    assert(points != NULL);
    //points = generatePoints(NUM_POINTS, NUM_DIMENSION, 0.0, 10.0);
    /*for (int i = 0; i < NUM_DIMENSION; i++)
    {
        for (int j = 0; j < NUM_POINTS; j++)
        {
            std::cout << points[i][j] << " ";
        }
        std::cout << std::endl;
    }
    */
    /* membership: the cluster id for each data object */
    membership = (int*)malloc(numPoints * sizeof(int));
    assert(membership != NULL);
    /* start the timer for the core computation -----------------------------*/
    start = std::chrono::high_resolution_clock::now();
    clusters = kMeansClustering(points, NUM_POINTS, NUM_DIMENSION, NUM_CLUSTERS, DELTA, membership, &loop_iterations);
    /* end the timer for the core computation -----------------------------*/
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CUDA kmeans: " << duration.count() << std::endl;

    /* output: the coordinates of the cluster centres ----------------------*/
    file_write("result.txt", NUM_CLUSTERS, NUM_POINTS, NUM_DIMENSION, clusters, membership);

    /* start the timer for the core computation -----------------------------*/
    start = std::chrono::high_resolution_clock::now();
    clusters = omp_kmeans(points, NUM_POINTS, NUM_DIMENSION, NUM_CLUSTERS, DELTA, membership);
    /* end the timer for the core computation -----------------------------*/
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "OMP kmeans: " << duration.count() << std::endl;
    /* output: the coordinates of the cluster centres ----------------------*/
    file_write("result_omp.txt", NUM_CLUSTERS, NUM_POINTS, NUM_DIMENSION, clusters, membership);

    start = std::chrono::high_resolution_clock::now();
    clusters = seq_kmeans(points, NUM_DIMENSION, NUM_POINTS, NUM_CLUSTERS, DELTA, membership, &loop_iterations);
    /* end the timer for the core computation -----------------------------*/
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "seq kmeans: " << duration.count() << std::endl;
    /* output: the coordinates of the cluster centres ----------------------*/
    free(points);
    file_write("result_seq.txt", NUM_CLUSTERS, NUM_POINTS, NUM_DIMENSION, clusters, membership);

    

    free(membership);
    free(clusters[0]);
    free(clusters);
}
