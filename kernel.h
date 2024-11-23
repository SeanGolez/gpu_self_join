#include "structs.h"
#include "params.h"

//original with unsigned long long for the counter (atomic)

//with long long unsigned int
__global__ void kernelBruteForce(unsigned int *N, unsigned int *debug1, unsigned int *debug2, DTYPE *epsilon, 
	unsigned long long int * cnt, DTYPE* database, int * pointIDKey, int * pointInDistVal);

__global__ void kernelNDGridIndexGlobal(unsigned int *debug1, unsigned int *debug2, unsigned int *N,  
	unsigned int * offset, DTYPE* database, DTYPE * epsilon, struct grid * allIndex, unsigned int * allIndexLookupArr, 
	struct gridCellLookup * allGridCellLookupArrStart, struct gridCellLookup * allGridCellLookupArrStartEnd, DTYPE* allMinArr, unsigned int * allNCells, 
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

__device__ void evaluateCell( unsigned int *N, unsigned int* allNCells, unsigned int* indexes, DTYPE* database, 
	DTYPE * epsilon, struct grid * allIndex, unsigned int * allIndexLookupArr, struct gridCellLookup * allGridCellLookupArrStart, struct gridCellLookup * allGridCellLookupArrStartEnd, 
	DTYPE* point, unsigned int* cnt,int* pointIDKey, int* pointInDistVal, int pointIdx, bool differentCell, unsigned int* nDCellIDs, CTYPE* workCounts);

__forceinline__ __device__ void evalPoint(unsigned int *N, unsigned int* allIndexLookupArr, int k, DTYPE* database, DTYPE * epsilon, DTYPE* point, 
	unsigned int* cnt, int* pointIDKey, int* pointInDistVal, int pointIdx, bool differentCell);

//functions for index on the GPU
__global__ void kernelIndexComputeNonemptyCells(DTYPE* database, unsigned int *N, DTYPE* epsilon, DTYPE* minArr, unsigned int * nCells, uint64_t * pointCellArr);
__global__ void kernelInitEnumerateDB(unsigned int * databaseVal, unsigned int *N);

__global__ void kernelIndexComputeAdjacentCells(uint64_t * cellDistCalcArr, uint64_t * uniqueCellArr, uint64_t * cellNumPointsArr, unsigned int * nCells, unsigned int * nNonEmptyCells, int *incrementors, unsigned int * nAdjCells);

__global__ void kernelMapPointToNumDistCalcs(uint64_t * pointDistCalcArr, DTYPE* database, unsigned int *N, DTYPE* epsilon, DTYPE* minArr, unsigned int * nCells, uint64_t *cellDistCalcArr,  uint64_t * uniqueCellArr, unsigned int * nNonEmptyCells);