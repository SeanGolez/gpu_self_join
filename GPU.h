#include "structs.h"
#include "params.h"



void makeDistanceTableGPUBruteForce(std::vector<std::vector <DTYPE> > * NDdataPoints, DTYPE* epsilon, struct table * neighborTable, unsigned long long int * totalNeighbors);

void distanceTableNDGridBatches(std::vector<std::vector<DTYPE> > * NDdataPoints, DTYPE * epsilon, struct grid * allIndex, 
	struct gridCellLookup * allGridCellLookupArr, unsigned int * allNNonEmptyCells, DTYPE* allMinArr, unsigned int * allNCells, 
	unsigned int * allIndexLookupArr, struct neighborTableLookup * neighborTable, std::vector<struct neighborDataPtrs> * pointersToNeighbors, 
	uint64_t * totalNeighbors, CTYPE* workCounts, unsigned int * orderedIndexPntIDs, std::vector<indexArrayPntGroups> * indexGroups, unsigned int * orderedQueryPntIDs,
	unsigned int * whichIndexPoints);


unsigned long long callGPUBatchEst(unsigned int DBSIZE, DTYPE* dev_database, DTYPE epsilon, unsigned int * whichIndexPoints, struct grid * dev_allGrids, 
	unsigned int * dev_allIndexLookupArr, struct gridCellLookup * dev_allGridCellLookupArr, unsigned int * gridIncrementEachIndex, DTYPE* dev_allMinArr, 
	unsigned int * dev_allNCells, unsigned int * allNNonEmptyCells, unsigned int * orderedQueryPntIDs, unsigned int * retNumBatches, unsigned int * retGPUBufferSize);

void constructNeighborTableKeyValueWithPtrs(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt);

void warmUpGPU();

void populateNDGridIndexAndLookupArrayGPU(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE *epsilon, DTYPE* minArr,  uint64_t totalCells, unsigned int * nCells, struct gridCellLookup ** gridCellLookupArr, struct grid ** index, unsigned int * indexLookupArr, unsigned int *nNonEmptyCells, std::vector<std::vector<int>> *incrementorVects, std::vector<workArrayPnt> *totalPointsWork);

//for the brute force version without batches
void constructNeighborTableKeyValue(int * pointIDKey, int * pointInDistValue, struct table * neighborTable, unsigned int * cnt);




//Unicomp requires multiple updates to the neighbortable for a given point
//This allows updating the neighbortable for the same point
void constructNeighborTableKeyValueWithPtrsWithMultipleUpdates(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt, pthread_mutex_t * pointLocks);

//Unicomp requires multiple updates to the neighbortable for a given point
//This allows updating the neighbortable for the same point
//WITHOUT VECTORS FOR DATA
void constructNeighborTableKeyValueWithPtrsWithMultipleUpdatesMultipleDataArrays(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt, int * uniqueKeys, int * uniqueKeyPosition, unsigned int numUniqueKeys);

//Unicomp requires multiple updates to the neighbortable for a given point
//This allows updating the neighbortable for the same point
//Without locks
void constructNeighborTableKeyValueWithPtrsBatchMaskArray(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt, int batchNum);

//Sort the queries by their workload based on the number of points in the cell
//From hybrid KNN paper in GPGPU'19 
void computeWorkDifficulty(unsigned int * outputOrderedQueryPntIDs, struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, unsigned int * indexLookupArr, struct grid * index);
