#include "kMeans.h"

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

float** seq_kmeans(float** objects,
    int     numCoords,
    int     numObjs,
    int     numClusters,
    float   threshold,
    int* membership,
    int* loop_iterations)
{
    int      i, j, index, loop = 0;
    int* newClusterSize;
    float    delta;
    float** clusters;
    float** newClusters;

    malloc2D(clusters, numCoords, numClusters, float);

    for (i = 0; i < numCoords; i++)
        for (j = 0; j < numClusters; j++)
            clusters[i][j] = objects[i][j];

    for (i = 0; i < numObjs; i++)
        membership[i] = -1;

    newClusterSize = (int*)calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);
    malloc2D(newClusters, numCoords, numClusters, float);

    do {
        delta = 0.0;
        for (i = 0; i < numObjs; i++) {
            index = find_nearest_cluster(numClusters, numCoords, numObjs, i, objects, clusters);

            if (membership[i] != index)
                delta += 1.0;

            membership[i] = index;

            newClusterSize[index]++;
            for (j = 0; j < numCoords; j++)
                newClusters[j][index] += objects[j][i];
        }

        for (i = 0; i < numClusters; i++) {
            for (j = 0; j < numCoords; j++) {
                if (newClusterSize[i] > 0)
                    clusters[j][i] = newClusters[j][i] / newClusterSize[i];
                newClusters[j][i] = 0.0;
            }
            newClusterSize[i] = 0;
        }

        delta /= numObjs;
    } while (delta > threshold && loop++ < 500);

    *loop_iterations = loop + 1;

    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return clusters;
}