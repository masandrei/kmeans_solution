#include <stdio.h>
#include <stdlib.h>

#include <omp.h>
#include "kmeans.h"


__inline static
float euclid_dist_2(int numDimension,
    float** data,
    float** clusters,
    int pointId,
    int clusterId)
{
    float ans = 0;
    for (int i = 0; i < numDimension; i++)
    {
        ans += (data[i][pointId] - clusters[i][clusterId]) * (data[i][pointId] - clusters[i][clusterId]);
    }
    return ans;
}

__inline static
int find_nearest_cluster(int numClusters,
    int numDimensions,
    int numPoints,
    int pointId,
    float** points,
    float** clusters)
{

    int index = 0;
    float min_dist = euclid_dist_2(numDimensions, points, clusters, pointId, 0);
    float dist;

    for (int i = 1; i < numClusters; i++) {
        dist = euclid_dist_2(numDimensions, points, clusters, pointId, i);
        if (dist < min_dist) {
            min_dist = dist;
            index = i;
        }
    }
    return(index);
}

float** omp_kmeans( float** points,
    int     numPoints,
    int     numDimensions,
    int     numClusters,
    float   threshold,
    int* membership)
{

    int      i, j, k, index, loop = 0;
    int* newClusterSize;
    float    delta;
    float** clusters;
    float** newClusters;

    int      nthreads;
    int** local_newClusterSize;
    float*** local_newClusters;

    nthreads = omp_get_max_threads();

    malloc2D(clusters, numDimensions, numClusters, float);
    matrixCopy(points, clusters, numDimensions, numClusters);
        

    for (i = 0; i < numPoints; i++)
    {
        membership[i] = -1;
    }

    newClusterSize = (int*)calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    malloc2D(newClusters, numDimensions, numClusters, float);
    malloc2D(local_newClusterSize, nthreads, numClusters, int);
       

    local_newClusters = (float***)malloc(nthreads * sizeof(float**));
    assert(local_newClusters != NULL);
    local_newClusters[0] = (float**)malloc(nthreads * numClusters * sizeof(float*));
    assert(local_newClusters[0] != NULL);
    for (i = 1; i < nthreads; i++)
    {
        local_newClusters[i] = local_newClusters[i - 1] + numClusters;
    }   
    for (i = 0; i < nthreads; i++) {
        for (j = 0; j < numDimensions; j++) 
        {
            local_newClusters[i][j] = (float*)calloc(numClusters, sizeof(float));
            assert(local_newClusters[i][j] != NULL);
        }
    }

    do {
        delta = 0.0;

        #pragma omp parallel shared(points,clusters,membership,local_newClusters,local_newClusterSize)
        {
            int tid = omp_get_thread_num();
            #pragma omp for private(i,j,index) firstprivate(numPoints,numClusters,numDimensions) schedule(static) reduction(+:delta)
            for (i = 0; i < numPoints; i++) {
                index = find_nearest_cluster(numClusters, numDimensions, numPoints, i, points, clusters);

                if (membership[i] != index)
                {
                    delta += 1.0;
                }

                membership[i] = index;

                local_newClusterSize[tid][index]++;
                for (j = 0; j < numDimensions; j++)
                {
                    local_newClusters[tid][j][index] += points[j][i];
                }
            }
        }

        for (i = 0; i < numClusters; i++) {
            for (j = 0; j < nthreads; j++) {
                newClusterSize[i] += local_newClusterSize[j][i];
                local_newClusterSize[j][i] = 0;
                for (k = 0; k < numDimensions; k++) {
                    newClusters[k][i] += local_newClusters[j][k][i];
                    local_newClusters[j][k][i] = 0.0;
                }
            }
        }

        for (i = 0; i < numClusters; i++) {
            for (j = 0; j < numDimensions; j++) {
                if (newClusterSize[i] > 0)
                {
                    clusters[j][i] = newClusters[j][i] / newClusterSize[i];
                }
                newClusters[j][i] = 0.0;
            }
            newClusterSize[i] = 0;
        }

        delta /= numPoints;
    } while (delta > threshold && loop++ < 500);

    free(local_newClusters);
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return clusters;
}