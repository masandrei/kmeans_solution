#include <fstream>
#include <stdio.h>
#include <iostream>
#include <string>

#include "kMeans.h"

using namespace std;

float** file_read(char* fileName, int* numPoints, int* numDimensions)
{
	ifstream inFile;
	inFile.open(fileName);

	if (!inFile.is_open())
	{
		cerr << "Could not open the file";
		exit(1);
	}
	inFile >> *numPoints; // read the number of points
	inFile >> *numDimensions; // read the number of dimensions

	float** resultingMatrix; // allocate a matrix to store the points
	malloc2D(resultingMatrix, *numDimensions, *numPoints, float);
	for (int i = 0; i < *numDimensions; i++)
	{
		for (int j = 0; j < *numPoints; j++)
		{
			inFile >> resultingMatrix[i][j]; //read coordinates of jth point in ith dimension
		}
	}
	inFile.close();
	return resultingMatrix;
}

void file_write(char* fileName, int numClusters, int numPoints, int numDimensions, float** clusterCentroids, int* membership)
{
	ofstream outFile;
	outFile.open(fileName);

	if (!outFile.is_open())
	{
		cerr << "Could not open the file";
		exit(1);
	}
	for (int i = 0; i < numDimensions; i++)
	{
		for (int j = 0; j < numClusters; j++)
		{
			outFile << clusterCentroids[i][j] << " ";
		}
		outFile << std::endl;
	}
	for (int i = 0; i < numPoints; i++)
	{
		outFile << membership[i] << " ";
	}
	outFile << std::endl;
	outFile.close();
}
