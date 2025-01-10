#include "structs.h"
#include "params.h"

//original with unsigned long long for the counter (atomic)

//with long long unsigned int
__global__ void kernelBruteForce(unsigned int *N, unsigned int *debug1, unsigned int *debug2, DTYPE *epsilon, 
	unsigned long long int * cnt, DTYPE* database, int * pointIDKey, int * pointInDistVal);

__global__ void kernelNDGridIndexGlobal(unsigned int *debug1, unsigned int *debug2, unsigned int *N,  
	unsigned int * offset, unsigned int * indexGroupOffset, DTYPE* database, DTYPE * epsilon, struct grid * index, unsigned int * indexLookupArr, 
	struct gridCellLookup * gridCellLookupArrStart, struct gridCellLookup * gridCellLookupArrStartEnd, DTYPE* minArr, unsigned int * nCells, 
	unsigned int * orderedIndexPntIDs, unsigned int * cnt, int * pointIDKey, int * pointInDistVal, CTYPE* workCounts);

__global__ void kernelNDGridIndexBatchEstimator(unsigned int *debug1, unsigned int *debug2, unsigned int *N,  
	unsigned int * sampleOffset, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, 
	struct gridCellLookup * gridCellLookupArrStart, struct gridCellLookup * gridCellLookupArrEnd,DTYPE* minArr, unsigned int * nCells, unsigned int * cnt,
	unsigned int * orderedQueryPntIDs);

__device__ uint64_t getLinearID_nDimensionsGPU(unsigned int * indexes, unsigned int * dimLen, unsigned int nDimensions);
__device__ void getNDimIndexesFromLinearIdxGPU(unsigned int * indexes, unsigned int * dimLen, unsigned int nDimensions, uint64_t linearId);
__device__ int binarySearch(uint64_t* d_array, unsigned int size, uint64_t value);

__global__ void kernelSortPointsInCells(DTYPE* database, struct grid * index, unsigned int* indexLookupArr, unsigned int nNonEmptyCells);

__global__ void kernelUniqueKeys(int * pointIDKey, unsigned int * N, int * uniqueKey, int * uniqueKeyPosition, unsigned int * cnt);

__device__ void evaluateCell(unsigned int* nCells, unsigned int* indexes, DTYPE* database, 
	DTYPE * epsilon, struct grid * index, unsigned int * indexLookupArr, struct gridCellLookup * gridCellLookupArrStart, struct gridCellLookup * gridCellLookupArrStartEnd, 
	DTYPE* point, unsigned int* cnt,int* pointIDKey, int* pointInDistVal, int pointIdx, bool differentCell, unsigned int* nDCellIDs, CTYPE* workCounts);


//functions for index on the GPU
__global__ void kernelIndexComputeNonemptyCells(DTYPE* database, unsigned int *N, DTYPE* epsilon, DTYPE* minArr, unsigned int * nCells, uint64_t * pointCellArr);
__global__ void kernelInitEnumerateDB(unsigned int * databaseVal, unsigned int *N);

__global__ void kernelIndexComputeAdjacentCells(uint64_t * cellDistCalcArr, uint64_t * uniqueCellArr, uint64_t * cellNumPointsArr, unsigned int * nCells, unsigned int * nNonEmptyCells, int *incrementors, unsigned int * nAdjCells);

__global__ void kernelMapPointToNumDistCalcs(uint64_t * pointDistCalcArr, DTYPE* database, unsigned int *N, DTYPE* epsilon, DTYPE* minArr, unsigned int * nCells, uint64_t *cellDistCalcArr,  uint64_t * uniqueCellArr, unsigned int * nNonEmptyCells);


//functions for generate histogram on the GPU
__global__ void transposeDataset(DTYPE * dataset, DTYPE * transposedDataset, const unsigned int NPOINTS);

__global__ void kernelEstimateAvgDistBruteForce(const unsigned int NPOINTS,
	unsigned long long int * cnt, DTYPE* database, DTYPE * total_distance);

//Transposed version of above
__global__ void kernelEstimateAvgDistBruteForce_T(const unsigned int NPOINTS, 
	unsigned long long int * cnt, DTYPE* database, DTYPE * total_distance);

//For ILP distance arrays
__device__ void zeroRunningDistArray(DTYPE * runningDist);

//From KNN-SJ
__global__ void kernelKDistBruteForce(const unsigned int NPOINTS, const unsigned int offset, 
	unsigned long long int * cnt, DTYPE* database, const DTYPE avg_distance, unsigned int * histogram, const DTYPE bucket_width);

__global__ void kernelKDistBruteForce_T(const unsigned int NPOINTS, const unsigned int offset, 
	unsigned long long int * cnt, DTYPE* database, const DTYPE avg_distance, unsigned int * histogram, const DTYPE bucket_width);
	
//Compute estimate on GPU from histogram
__global__ void computeSelectivityEstimateFromHistogram(const DTYPE epsilon, unsigned int * histogram, const DTYPE bucket_width, unsigned long long int * estimatedTotalResultSetSize);
