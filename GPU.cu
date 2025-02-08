#include <pthread.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "structs.h"
#include <stdio.h>
#include "kernel.h"
#include <math.h>
#include "GPU.h"
#include <algorithm>
#include "omp.h"
#include <queue>
#include <unistd.h>
#include <random>

//thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h> //for streams for thrust (added with Thrust v1.8)


//for warming up GPU:
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>



using namespace std;

//Error checking GPU calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//sort descending
bool compareWorkArrayByNumPointsInCell(const workArray &a, const workArray &b)
{
    return a.pntsInCell > b.pntsInCell;
}



//sort ascending
bool compareByPointValue(const key_val_sort &a, const key_val_sort &b)
{
    return a.value_at_dim < b.value_at_dim;
}


unsigned long long callGPUBatchEst(unsigned int DBSIZE, DTYPE* dev_database, DTYPE epsilon, unsigned int * dev_whichIndexPoints, struct grid * dev_grid, 
	unsigned int * dev_indexLookupArr, struct gridCellLookup * dev_gridCellLookupArr, DTYPE* dev_minArr, unsigned int * dev_nCells, unsigned int * dev_nNonEmptyCells, 
	unsigned int * dev_orderedQueryPntIDs, gridCellLookup ** dev_startGridPtrs, gridCellLookup ** dev_stopGridPtrs, grid ** dev_startIndexPtrs,
	unsigned int * retNumBatches, unsigned int * retGPUBufferSize)
{



	//CUDA error code:
	cudaError_t errCode;

	printf("\n\n***********************************\nEstimating Batches:");
	cout<<"\n** BATCH ESTIMATOR: Sometimes the GPU will error on a previous execution and you won't know. \nLast error start of function: "<<cudaGetLastError();



//////////////////////////////////////////////////////////
	//ESTIMATE THE BUFFER SIZE AND NUMBER OF BATCHES ETC BY COUNTING THE NUMBER OF RESULTS
	//TAKE A SAMPLE OF THE DATA POINTS, NOT ALL OF THEM
	//Use sampleRate for this
	/////////////////////////////////////////////////////////

	
	// printf("\nDon't estimate: calculate the entire thing (for testing)");
	//Parameters for the batch size estimation.
	double sampleRate=SAMPLERATE; //sample 1.5% of the points in the dataset sampleRate=0.01. 
						//Sample the entire dataset(no sampling) sampleRate=1
						//0.015- used in Journal version of high-D paper for reorderqueries
						//0.01- used for all otherpapers (HPBDC, JPDC, DaMoN)
	int offsetRate=1.0/sampleRate;
	printf("\nOffset: %d", offsetRate);



	/////////////////
	//N GPU threads
	////////////////

	unsigned int * dev_N_batchEst; 
	dev_N_batchEst=(unsigned int*)malloc(sizeof(unsigned int));

	unsigned int * N_batchEst; 
	N_batchEst=(unsigned int*)malloc(sizeof(unsigned int));
	*N_batchEst=DBSIZE*sampleRate;


	//allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_N_batchEst, sizeof(unsigned int)));
	
	//copy N to device 
	//N IS THE NUMBER OF THREADS
	gpuErrchk(cudaMemcpy( dev_N_batchEst, N_batchEst, sizeof(unsigned int), cudaMemcpyHostToDevice));
	
	/////////////
	//count the result set size 
	////////////

	unsigned int * dev_cnt_batchEst; 
	dev_cnt_batchEst=(unsigned int*)malloc(sizeof(unsigned int) * NUMRANDINDEXES * NUMRANDROTATIONS);

	unsigned int * cnt_batchEst; 
	cnt_batchEst=(unsigned int*)malloc(sizeof(unsigned int) * NUMRANDINDEXES * NUMRANDROTATIONS);
	for( unsigned int i=0; i<NUMRANDINDEXES * NUMRANDROTATIONS; i++) {
		cnt_batchEst[i]=0;
	}
	


	//allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_cnt_batchEst, sizeof(unsigned int) * NUMRANDINDEXES * NUMRANDROTATIONS));

	//copy cnt to device 
	gpuErrchk(cudaMemcpy( dev_cnt_batchEst, cnt_batchEst, sizeof(unsigned int) * NUMRANDINDEXES * NUMRANDROTATIONS, cudaMemcpyHostToDevice));

	//////////////////
	//SAMPLE OFFSET - TO SAMPLE THE DATA TO ESTIMATE THE TOTAL NUMBER OF KEY VALUE PAIRS
	/////////////////

	//offset into the database when batching the results
	unsigned int * sampleOffset; 
	sampleOffset=(unsigned int*)malloc(sizeof(unsigned int));
	*sampleOffset=offsetRate;


	unsigned int * dev_sampleOffset; 
	dev_sampleOffset=(unsigned int*)malloc(sizeof(unsigned int));

	//allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_sampleOffset, sizeof(unsigned int)));

	//copy offset to device 
	gpuErrchk(cudaMemcpy( dev_sampleOffset, sampleOffset, sizeof(unsigned int), cudaMemcpyHostToDevice));

	////////////////////////////////////
	//TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////			

	//debug values
	unsigned int * dev_debug1; 
	dev_debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug1=0;

	unsigned int * dev_debug2; 
	dev_debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug2=0;

	unsigned int * debug1; 
	debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug1=0;

	unsigned int * debug2; 
	debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug2=0;



	//allocate on the device
	gpuErrchk(cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int )));
	
	gpuErrchk(cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int )));
	
	//copy debug to device
	gpuErrchk(cudaMemcpy( dev_debug1, debug1, sizeof(unsigned int), cudaMemcpyHostToDevice ));
	

	gpuErrchk(cudaMemcpy( dev_debug2, debug2, sizeof(unsigned int), cudaMemcpyHostToDevice ));
	
	////////////////////////////////////
	//END TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////




	const int TOTALBLOCKSBATCHEST=ceil((1.0*(DBSIZE)*sampleRate)/(1.0*BLOCKSIZE));
	printf("\ntotal blocks: %d",TOTALBLOCKSBATCHEST);

	

	kernelNDGridIndexBatchEstimator<<< TOTALBLOCKSBATCHEST, BLOCKSIZE>>>(dev_debug1, dev_debug2, dev_N_batchEst, 
		dev_sampleOffset, DBSIZE, dev_database, epsilon, dev_whichIndexPoints, dev_grid, dev_indexLookupArr, 
		dev_gridCellLookupArr, dev_minArr, dev_nCells, dev_startGridPtrs, dev_stopGridPtrs, dev_startIndexPtrs,
		dev_cnt_batchEst, dev_nNonEmptyCells, dev_orderedQueryPntIDs);
		cout<<"\n** ERROR FROM KERNEL LAUNCH OF BATCH ESTIMATOR: "<<cudaGetLastError();
		// find the size of the number of results
		errCode=cudaMemcpy( cnt_batchEst, dev_cnt_batchEst, sizeof(unsigned int) * NUMRANDINDEXES * NUMRANDROTATIONS, cudaMemcpyDeviceToHost);
		if(errCode != cudaSuccess) {
		cout << "\nError: getting cnt for batch estimate from GPU Got error with code " << errCode << endl; 
		}
		else
		{
			for( unsigned int i=0; i<NUMRANDINDEXES * NUMRANDROTATIONS; i++ ) {
				printf("\nGPU: result set size for estimating the number of batches (sampled) for index %d: %u", i, cnt_batchEst[i]);
			}
			
		}

	#ifndef PYTHON	
	double alpha=0.05; //overestimation factor
	#endif

	#ifdef PYTHON
	double alpha=0.10;
	#endif

	unsigned int largestGPUBufferSize=GPUBUFFERSIZE;
	uint64_t estimatedTotalSizeWithAlpha = 0;

	unsigned int batchSum = 0;

	printf("\n");
	for( unsigned int i=0; i<NUMRANDINDEXES * NUMRANDROTATIONS; i++ ) {
		printf("\nIndex %d batches", i);
		printf("\n================");

		unsigned int GPUBufferSize=GPUBUFFERSIZE;

		#if COUNTMETRICS==1	
		gpuErrchk(cudaMemcpy( debug1, dev_debug1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		printf("\nGPU: sampled number of cells visited: %u",*debug1);
		#endif	

		uint64_t estimatedNeighbors=(uint64_t)cnt_batchEst[i]*(uint64_t)offsetRate;	
		printf("\nFrom gpu cnt: %d, offset rate: %d", cnt_batchEst[i],offsetRate);
		
		
		


		
		uint64_t estimatedTotalSizeWithAlphaSingleIndex = estimatedNeighbors*(1.0+alpha*1.0);
		printf("\nEstimated result set size: %lu", estimatedNeighbors);
		printf("\nEstimated result set size (with Alpha %f): %lu", alpha,estimatedTotalSizeWithAlphaSingleIndex);
		estimatedTotalSizeWithAlpha+=estimatedTotalSizeWithAlphaSingleIndex;	
		


		if (estimatedNeighbors<(GPUBufferSize*GPUSTREAMS))
		{
			printf("\nSmall buffer size, increasing alpha to: %f",alpha*3.0);
			GPUBufferSize=estimatedNeighbors*(1.0+(alpha*3.0))/(GPUSTREAMS);		//we do 3*alpha for small datasets because the
																			//sampling will be worse for small datasets
																			//but we fix the number of streams.			
		}

		unsigned int numBatches=ceil(((1.0+alpha)*estimatedNeighbors*1.0)/((uint64_t)GPUBufferSize*1.0));

		// use largest buffer size
		if( GPUBufferSize > largestGPUBufferSize) {
			largestGPUBufferSize = GPUBufferSize;
		}

		retNumBatches[i] = numBatches;
		batchSum += numBatches;

		printf("\nNumber of batches %d, buffer size: %d", numBatches, GPUBufferSize);

		printf("\n");
	}

	//Make sure at least MINBATCHES are executed to mitigate against large pinned memory allocation overheads
	while (batchSum<MINBATCHES)
	{
		unsigned int smallestBatchNum = UINT_MAX;
		unsigned int * smallestRetNumBatchesPtr = retNumBatches;
		for( unsigned int i=0; i<NUMRANDINDEXES*NUMRANDROTATIONS; i++ )
		{
			if( retNumBatches[i] < smallestBatchNum )
			{
				smallestBatchNum = retNumBatches[i];
				smallestRetNumBatchesPtr = &retNumBatches[i];
			}
		}
		(*smallestRetNumBatchesPtr)++;
	}

	*retGPUBufferSize=largestGPUBufferSize;

	printf("\nBuffer size: %d", largestGPUBufferSize);
	printf("\nEstimated total result set size (with Alpha %f): %lu", alpha,estimatedTotalSizeWithAlpha);

	printf("\nEnd Batch Estimator\n***********************************\n");




	cudaFree(dev_cnt_batchEst);	
	cudaFree(dev_N_batchEst);
	cudaFree(dev_sampleOffset);
	
return estimatedTotalSizeWithAlpha;

}

double distanceTableNDGridBatches(std::vector<std::vector<std::vector<DTYPE>>> * allRotatedNDdataPoints, DTYPE * epsilon, struct grid * allIndex, 
	struct gridCellLookup * allGridCellLookupArr, unsigned int * allNNonEmptyCells, DTYPE* allMinArr, unsigned int * allNCells, 
	unsigned int * allIndexLookupArr, struct neighborTableLookup * neighborTable, std::vector<struct neighborDataPtrs> * pointersToNeighbors, 
	uint64_t * totalNeighbors, CTYPE* workCounts, unsigned int * orderedIndexPntIDs, std::vector<indexArrayPntGroups> * indexGroups, unsigned int * orderedQueryPntIDs,
	unsigned int * whichIndexPoints)
{
	// create total num empty cells for all indexes variable
	unsigned int totalNNonemptyCells = 0;
	for( int i=0; i<NUMRANDINDEXES * NUMRANDROTATIONS; i ++)
	{
		totalNNonemptyCells += allNNonEmptyCells[i];
	}

	unsigned int DBSIZE_var = (*allRotatedNDdataPoints)[0].size();
	unsigned int * DBSIZE = &DBSIZE_var;


	double tKernelResultsStart=omp_get_wtime();
	
	cout<<"\n** Sometimes the GPU will error on a previous execution and you won't know. \nLast error start of function: "<<cudaGetLastError();

	//CUDA error code:
	cudaError_t errCode;

	unsigned int * dev_orderedQueryPntIDs=NULL;

	//Reordering the query points based on work
	#if QUERYREORDER==1
	// note: commented this out because orderQueryPntIDs created before function call
	// unsigned int * orderedQueryPntIDs=new unsigned int[NDdataPoints->size()];
	// computeWorkDifficulty(orderedQueryPntIDs, gridCellLookupArr, nNonEmptyCells, indexLookupArr, index);
	//allocate memory on device:
	gpuErrchk(cudaMalloc( (void**)&dev_orderedQueryPntIDs, sizeof(unsigned int)*(*DBSIZE)));
	gpuErrchk(cudaMemcpy(dev_orderedQueryPntIDs, orderedQueryPntIDs, sizeof(unsigned int)*(*DBSIZE), cudaMemcpyHostToDevice));

	#endif

	unsigned int * dev_orderedIndexPntIDs=NULL;
	gpuErrchk(cudaMalloc( (void**)&dev_orderedIndexPntIDs, sizeof(unsigned int)*(*DBSIZE)));
	gpuErrchk(cudaMemcpy(dev_orderedIndexPntIDs, orderedIndexPntIDs, sizeof(unsigned int)*(*DBSIZE), cudaMemcpyHostToDevice));




	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////

	// TODO: use const unsigned int DBSIZE, then dont need to cudaMalloc, same for epsilon
	
	printf("\nIn main GPU method: DBSIZE is: %u",*DBSIZE);cout.flush();
	
	DTYPE* database = (DTYPE*)malloc(sizeof(DTYPE)*(*DBSIZE)*(GPUNUMDIM)*(NUMRANDROTATIONS));  
	DTYPE* dev_database;
	
	//allocate memory on device:
	gpuErrchk(cudaMalloc( (void**)&dev_database, sizeof(DTYPE)*(GPUNUMDIM)*(*DBSIZE)*(NUMRANDROTATIONS)));

	//copy the database from the ND vector to the array:
	for( int i=0; i<NUMRANDROTATIONS; i++ ) {
		for (int j=0; j<(*DBSIZE); j++){
			std::copy((*allRotatedNDdataPoints)[i][j].begin(), (*allRotatedNDdataPoints)[i][j].end(), database+((*DBSIZE)*GPUNUMDIM*i)+(j*(GPUNUMDIM)));
		}
	}


	//copy database to the device
	gpuErrchk(cudaMemcpy(dev_database, database, sizeof(DTYPE)*(GPUNUMDIM)*(*DBSIZE)*(NUMRANDROTATIONS), cudaMemcpyHostToDevice));

	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////




	///////////////////////////////////
	//COPY THE INDEX TO THE GPU
	///////////////////////////////////

	struct grid * dev_allGrids;
	//allocate memory on device:
	gpuErrchk(cudaMalloc( (void**)&dev_allGrids, sizeof(struct grid)*(totalNNonemptyCells)));
	
	//copy grid index to the device:
	gpuErrchk(cudaMemcpy(dev_allGrids, allIndex, sizeof(struct grid)*(totalNNonemptyCells), cudaMemcpyHostToDevice));

	printf("\nSize of index sent to GPU (MiB): %f", (DTYPE)sizeof(struct grid)*(totalNNonemptyCells)/(1024.0*1024.0));


	///////////////////////////////////
	//END COPY THE INDEX TO THE GPU
	///////////////////////////////////



	///////////////////////////////////
	//COPY THE LOOKUP ARRAY TO THE DATA ELEMS TO THE GPU
	///////////////////////////////////

	#if SORT==1
	printf("\nSORTING ALL DIMENSIONS FOR VARIANCE NOW");
	printf("\nSORTIDU USES THE FIRST UNINXEDED DIMENSION");
	
	double tstartsort=omp_get_wtime();
	
	int sortDim=0;
	struct key_val_sort tmp;
	std::vector<struct key_val_sort> tmp_to_sort;
	unsigned int totalLength=0;

	if(GPUNUMDIM > NUMINDEXEDDIM)
		sortDim = NUMINDEXEDDIM;

	for (int i=0; i<(*nNonEmptyCells); i++)
	// for (int i=0; i<1; i++)
	{
		// if(index[i].indexmin < index[i].indexmax){
			// printf("Size cell: %d, %d\n", i,(index[i].indexmax-index[i].indexmin)+1);
			for (int j=0; j<(index[i].indexmax-index[i].indexmin)+1; j++)
			{
			unsigned int idx=index[i].indexmin+j;	
			tmp.pid=indexLookupArr[idx];
			tmp.value_at_dim=database[indexLookupArr[idx]*GPUNUMDIM+sortDim];
			tmp_to_sort.push_back(tmp);
			}

			
			totalLength+=tmp_to_sort.size();
			std::sort(tmp_to_sort.begin(),tmp_to_sort.end(),compareByPointValue); 

			//copy the sorted elements into the lookup array
			for (int x=0; x<tmp_to_sort.size(); x++)
			{
				indexLookupArr[index[i].indexmin+x]=tmp_to_sort[x].pid;
			}

			tmp_to_sort.clear();	
	}	

	double tendsort=omp_get_wtime();
	printf("\nSORT cells time (on host): %f", tendsort - tstartsort);

	// printf("\nTotal length: %u",totalLength);
	#endif
	
	unsigned int * dev_allIndexLookupArr;

	//allocate memory on device:
	gpuErrchk(cudaMalloc( (void**)&dev_allIndexLookupArr, sizeof(unsigned int)*(*DBSIZE)*(NUMRANDINDEXES)*(NUMRANDROTATIONS)));

	//copy lookup array to the device:
	gpuErrchk(cudaMemcpy(dev_allIndexLookupArr, allIndexLookupArr, sizeof(unsigned int)*(*DBSIZE)*(NUMRANDINDEXES)*(NUMRANDROTATIONS), cudaMemcpyHostToDevice));
	
	///////////////////////////////////
	//END COPY THE LOOKUP ARRAY TO THE DATA ELEMS TO THE GPU
	///////////////////////////////////





	///////////////////////////////////
	//COPY THE GRID CELL LOOKUP ARRAY 
	///////////////////////////////////

	
	
						
	struct gridCellLookup * dev_allGridCellLookupArr;

	//allocate memory on device:
	gpuErrchk(cudaMalloc( (void**)&dev_allGridCellLookupArr, sizeof(struct gridCellLookup)*(totalNNonemptyCells)));

	//copy lookup array to the device:
	gpuErrchk(cudaMemcpy(dev_allGridCellLookupArr, allGridCellLookupArr, sizeof(struct gridCellLookup)*(totalNNonemptyCells), cudaMemcpyHostToDevice));

	///////////////////////////////////
	//END COPY THE GRID CELL LOOKUP ARRAY 
	///////////////////////////////////


	
	///////////////////////////////////
	//COPY GRID DIMENSIONS TO THE GPU
	//THIS INCLUDES THE NUMBER OF CELLS IN EACH DIMENSION, 
	//AND THE STARTING POINT OF THE GRID IN THE DIMENSIONS 
	///////////////////////////////////

	//minimum boundary of the grid:
	DTYPE* dev_allMinArr;
	//Allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_allMinArr, sizeof(DTYPE)*(NUMINDEXEDDIM)*(NUMRANDINDEXES)*(NUMRANDROTATIONS)));
	
	gpuErrchk(cudaMemcpy( dev_allMinArr, allMinArr, sizeof(DTYPE)*(NUMINDEXEDDIM)*(NUMRANDINDEXES)*(NUMRANDROTATIONS), cudaMemcpyHostToDevice ));

	//number of cells in each dimension
	unsigned int * dev_allNCells;

	//Allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_allNCells, sizeof(unsigned int)*(NUMINDEXEDDIM)*(NUMRANDINDEXES)*(NUMRANDROTATIONS)));

	gpuErrchk(cudaMemcpy( dev_allNCells, allNCells, sizeof(unsigned int)*(NUMINDEXEDDIM)*(NUMRANDINDEXES)*(NUMRANDROTATIONS), cudaMemcpyHostToDevice ));

	///////////////////////////////////
	//END COPY GRID DIMENSIONS TO THE GPU
	///////////////////////////////////




	///////////////////////////////////
	//COUNT VALUES -- RESULT SET SIZE FOR EACH KERNEL INVOCATION
	///////////////////////////////////

	//total size of the result set as it's batched
	//this isnt sent to the GPU
	unsigned int * totalResultSetCnt;
	totalResultSetCnt=(unsigned int*)malloc(sizeof(unsigned int));
	*totalResultSetCnt=0;

	//count values - for an individual kernel launch
	//need different count values for each stream
	unsigned int * cnt;
	cnt=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	*cnt=0;

	unsigned int * dev_cnt; 
	

	//allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_cnt, sizeof(unsigned int)*GPUSTREAMS));

	///////////////////////////////////
	//END COUNT VALUES -- RESULT SET SIZE FOR EACH KERNEL INVOCATION
	///////////////////////////////////
	
	
	///////////////////////////////////
	//EPSILON
	///////////////////////////////////
	DTYPE* dev_epsilon;
	

	//Allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_epsilon, sizeof(DTYPE)));

	//copy to device
	gpuErrchk(cudaMemcpy( dev_epsilon, epsilon, sizeof(DTYPE), cudaMemcpyHostToDevice ));
	
	///////////////////////////////////
	//END EPSILON
	///////////////////////////////////


	///////////////////////////////////
	//NUMBER OF NON-EMPTY CELLS (might remove this later)
	///////////////////////////////////
	unsigned int * dev_allNNonEmptyCells;
	
	//Allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_allNNonEmptyCells, sizeof(unsigned int)*(NUMRANDINDEXES)*(NUMRANDROTATIONS)));
	//copy to device
	gpuErrchk(cudaMemcpy( dev_allNNonEmptyCells, allNNonEmptyCells, sizeof(unsigned int)*(NUMRANDINDEXES)*(NUMRANDROTATIONS), cudaMemcpyHostToDevice ));

	///////////////////////////////////
	//NUMBER OF NON-EMPTY CELLS
	///////////////////////////////////

	///////////////////////////////////
	//WHICH INDEX TO USE FOR EACH POINT
	///////////////////////////////////
	unsigned int * dev_whichIndexPoints;
	
	//Allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_whichIndexPoints, sizeof(unsigned int)*(*DBSIZE)));
	//copy to device
	gpuErrchk(cudaMemcpy( dev_whichIndexPoints, whichIndexPoints, sizeof(unsigned int)*(*DBSIZE), cudaMemcpyHostToDevice ));

	///////////////////////////////////
	//WHICH INDEX TO USE FOR EACH POINT
	///////////////////////////////////

	

	/*
	//////////////////////////////////
	//ND MASK -- The array, the offsets, and the size of the array
	//////////////////////////////////

	//NDMASK
	unsigned int * dev_gridCellNDMask;

	//Allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_gridCellNDMask, sizeof(unsigned int)*(*nNDMaskElems)));
	
	gpuErrchk(cudaMemcpy( dev_gridCellNDMask, gridCellNDMask, sizeof(unsigned int)*(*nNDMaskElems), cudaMemcpyHostToDevice ));
	
	//NDMASKOFFSETS
	unsigned int * dev_gridCellNDMaskOffsets;

	//Allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_gridCellNDMaskOffsets, sizeof(unsigned int)*(2*NUMINDEXEDDIM)));

	gpuErrchk(cudaMemcpy( dev_gridCellNDMaskOffsets, gridCellNDMaskOffsets, sizeof(unsigned int)*(2*NUMINDEXEDDIM), cudaMemcpyHostToDevice ));

	//////////////////////////////////
	//End ND MASK -- The array, the offsets, and the size of the array
	//////////////////////////////////
	*/

	//////////////////////////////////
	// find start and stop positions for each index for batch estimator
	//////////////////////////////////
	gridCellLookup ** startGridPtrs;
	startGridPtrs = (gridCellLookup **)malloc(sizeof(gridCellLookup *) * NUMRANDINDEXES * NUMRANDROTATIONS);
	gridCellLookup ** stopGridPtrs;
	stopGridPtrs = (gridCellLookup **)malloc(sizeof(gridCellLookup *) * NUMRANDINDEXES * NUMRANDROTATIONS);
	grid ** startIndexPtrs;
	startIndexPtrs = (grid **)malloc(sizeof(grid *) * NUMRANDINDEXES * NUMRANDROTATIONS);

	gridCellLookup * startGridPtr = dev_allGridCellLookupArr;
	grid * startIndexPtr = dev_allGrids;
	
	for(int i = 0; i<NUMRANDINDEXES * NUMRANDROTATIONS; i++)
	{
		startGridPtrs[i] = startGridPtr;
		startIndexPtrs[i] = startIndexPtr;

		startGridPtr += allNNonEmptyCells[i];
		startIndexPtr += allNNonEmptyCells[i];

		stopGridPtrs[i] = startGridPtr;
	}

	gridCellLookup ** dev_startGridPtrs;
	gpuErrchk(cudaMalloc((void**)&dev_startGridPtrs, sizeof(gridCellLookup *) * NUMRANDINDEXES * NUMRANDROTATIONS));
	gpuErrchk(cudaMemcpy( dev_startGridPtrs, startGridPtrs, sizeof(gridCellLookup *) * NUMRANDINDEXES * NUMRANDROTATIONS, cudaMemcpyHostToDevice ));

	gridCellLookup ** dev_stopGridPtrs;
	gpuErrchk(cudaMalloc((void**)&dev_stopGridPtrs, sizeof(gridCellLookup *) * NUMRANDINDEXES * NUMRANDROTATIONS));
	gpuErrchk(cudaMemcpy( dev_stopGridPtrs,stopGridPtrs, sizeof(gridCellLookup *) * NUMRANDINDEXES * NUMRANDROTATIONS, cudaMemcpyHostToDevice ));

	grid ** dev_startIndexPtrs;
	gpuErrchk(cudaMalloc((void**)&dev_startIndexPtrs, sizeof(grid *) * NUMRANDINDEXES * NUMRANDROTATIONS));
	gpuErrchk(cudaMemcpy( dev_startIndexPtrs, startIndexPtrs, sizeof(grid *) * NUMRANDINDEXES * NUMRANDROTATIONS, cudaMemcpyHostToDevice ));

	//////////////////////////////////
	// End find start and stop positions for each index for batch estimator
	//////////////////////////////////


	unsigned long long estimatedNeighbors=0;	
	unsigned int * numBatchesEachIndex = (unsigned int *)malloc(sizeof(unsigned int) * NUMRANDINDEXES * NUMRANDROTATIONS);
	unsigned int GPUBufferSize=0;

	double tstartbatchest=omp_get_wtime();
	estimatedNeighbors=callGPUBatchEst(*DBSIZE, dev_database, *epsilon, dev_whichIndexPoints, dev_allGrids, dev_allIndexLookupArr, dev_allGridCellLookupArr, 
										dev_allMinArr, dev_allNCells, dev_allNNonEmptyCells, dev_orderedQueryPntIDs, dev_startGridPtrs, dev_stopGridPtrs, dev_startIndexPtrs, 
										numBatchesEachIndex, &GPUBufferSize);	
	double tendbatchest=omp_get_wtime();
	printf("\nTime to estimate batches: %f",tendbatchest - tstartbatchest);
	// printf("\nIn Calling fn: Estimated neighbors: %llu, num. batches: %d, GPU Buffer size: %d",estimatedNeighbors, numBatches,GPUBufferSize);
	printf("\nIn Calling fn: Estimated neighbors: %llu, GPU Buffer size: %d", estimatedNeighbors ,GPUBufferSize);
	
	// Find largest number of batches
	unsigned int largestNumBatches = 0;
	unsigned int totalNumBatches = 0;
	for( int i=0; i<NUMRANDINDEXES * NUMRANDROTATIONS; i++ ) {
		// numBatchesEachIndex[i] = numBatches * batchDivider[i];
		
		// printf("\nAdjusted numBatches for index %d: %d", i, numBatchesEachIndex[i]);
		
		if( numBatchesEachIndex[i] > largestNumBatches ) {
			largestNumBatches = numBatchesEachIndex[i];
		}

		printf("\nNumber batches for index %d: %d", i, numBatchesEachIndex[i]);
		totalNumBatches += numBatchesEachIndex[i];
	}
	printf("\nTotal number of batches: %d", totalNumBatches);
	printf("\n");

	cudaFree(dev_whichIndexPoints);
	cudaFree(dev_startGridPtrs);
	cudaFree(dev_stopGridPtrs);
	cudaFree(dev_startIndexPtrs);
	free(startGridPtrs);
	free(stopGridPtrs);
	free(startIndexPtrs);



	//initialize new neighbortable. resize to the number of batches	
	//Only use this if using unicomp
	#if STAMP==1
	double tstartinitneighbortable=omp_get_wtime();
	for (int i=0; i<NDdataPoints->size(); i++)
	{
	neighborTable[i].cntNDataArrays=0;
	neighborTable[i].vectindexmin.resize(numBatches);
	neighborTable[i].vectindexmax.resize(numBatches);
	neighborTable[i].vectdataPtr.resize(numBatches);
	pthread_mutex_init(&neighborTable[i].pointLock,NULL);
	}
	double tendinitneighbortable=omp_get_wtime();
	printf("\nTime to init neighbortable (tid: %d): %f",  omp_get_thread_num(), tendinitneighbortable - tstartinitneighbortable);
	#endif

	/////////////////////////////////////////////////////////	
	//END BATCH ESTIMATOR	
	/////////////////////////////////////////////////////////

	////////////////////////////////////
	//NUMBER OF THREADS PER GPU STREAM
	////////////////////////////////////

	//THE NUMBER OF THREADS THAT ARE LAUNCHED IN A SINGLE KERNEL INVOCATION
	//CAN BE FEWER THAN THE NUMBER OF ELEMENTS IN THE DATABASE IF MORE THAN 1 BATCH
	unsigned int * N;
	N=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	
	unsigned int * dev_N; 

	//allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_N, sizeof(unsigned int)*GPUSTREAMS));
	
	////////////////////////////////////
	//NUMBER OF THREADS PER GPU STREAM
	////////////////////////////////////


	////////////////////////////////////
	//OFFSET INTO THE DATABASE FOR BATCHING THE RESULTS
	//BATCH NUMBER 
	////////////////////////////////////
	unsigned int * batchOffset; 
	batchOffset=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	
	unsigned int * dev_offset; 
	

	//allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_offset, sizeof(unsigned int)*GPUSTREAMS));

	/*
	//Batch number to calculate the point to process (in conjunction with the offset)
	//offset into the database when batching the results
	unsigned int * batchNumber; 
	batchNumber=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	unsigned int * dev_batchNumber; 

	//allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_batchNumber, sizeof(unsigned int)*GPUSTREAMS));

	*/

	////////////////////////////////////
	//END OFFSET INTO THE DATABASE FOR BATCHING THE RESULTS
	//BATCH NUMBER
	////////////////////////////////////
	

	////////////////////////////////////
	//TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////			

	//debug values
	unsigned int * dev_debug1; 

	unsigned int * dev_debug2; 

	unsigned int * debug1; 
	debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug1=0;

	unsigned int * debug2; 
	debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug2=0;

	//allocate on the device
	gpuErrchk(cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int ) ));
	
	gpuErrchk(cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int ) ));
	
	//copy debug to device
	gpuErrchk(cudaMemcpy( dev_debug1, debug1, sizeof(unsigned int), cudaMemcpyHostToDevice ));

	gpuErrchk(cudaMemcpy( dev_debug2, debug2, sizeof(unsigned int), cudaMemcpyHostToDevice ));


	////////////////////////////////////
	//END TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////			

	

	///////////////////
	//ALLOCATE POINTERS TO INTEGER ARRAYS FOR THE VALUES FOR THE NEIGHBORTABLES
	///////////////////

	//THE NUMBER OF POINTERS IS EQUAL TO THE NUMBER OF BATCHES
	for (int i=0; i<largestNumBatches; i++){
		
		struct neighborDataPtrs tmpStruct;
		tmpStruct.dataPtr=NULL;
		tmpStruct.sizeOfDataArr=0;
		
		pointersToNeighbors->push_back(tmpStruct);
	}

	///////////////////
	//END ALLOCATE POINTERS TO INTEGER ARRAYS FOR THE VALUES FOR THE NEIGHBORTABLES
	///////////////////



	///////////////////////////////////
	//ALLOCATE MEMORY FOR THE RESULT SET USING THE BATCH ESTIMATOR
	///////////////////////////////////

	int * dev_pointIDKey[GPUSTREAMS]; //key
	int * dev_pointInDistValue[GPUSTREAMS]; //value
	
	for (int i=0; i<GPUSTREAMS; i++)
	{
		gpuErrchk(cudaMalloc((void **)&dev_pointIDKey[i], sizeof(int)*GPUBufferSize));
		gpuErrchk(cudaMalloc((void **)&dev_pointInDistValue[i], sizeof(int)*GPUBufferSize));
	}	

	//HOST RESULT ALLOCATION FOR THE GPU TO COPY THE DATA INTO A PINNED MEMORY ALLOCATION
	//ON THE HOST
	//pinned result set memory for the host
	//the number of elements are recorded for that batch in resultElemCountPerBatch
	//NEED PINNED MEMORY ALSO BECAUSE YOU NEED IT TO USE STREAMS IN THRUST FOR THE MEMCOPY OF THE SORTED RESULTS	

	//PINNED MEMORY TO COPY FROM THE GPU	
	int * pointIDKey[GPUSTREAMS]; //key
	int * pointInDistValue[GPUSTREAMS]; //value
		
	double tstartpinnedresults=omp_get_wtime();
	
	for (int i=0; i<GPUSTREAMS; i++)
	{
	cudaMallocHost((void **) &pointIDKey[i], sizeof(int)*GPUBufferSize);
	cudaMallocHost((void **) &pointInDistValue[i], sizeof(int)*GPUBufferSize);
	}

	double tendpinnedresults=omp_get_wtime();
	printf("\nTime to allocate pinned memory for results (tid: %d): %f", omp_get_thread_num(), tendpinnedresults - tstartpinnedresults);
	printf("\nmemory requested for results ON GPU (GiB): %f",(double)(sizeof(int)*2*GPUBufferSize*GPUSTREAMS)/(1024*1024*1024));
	printf("\nmemory requested for results in MAIN MEMORY (GiB): %f",(double)(sizeof(int)*2*GPUBufferSize*GPUSTREAMS)/(1024*1024*1024));

	
	///////////////////////////////////
	//END ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////
	

	/////////////////////////////////
	//SET OPENMP ENVIRONMENT VARIABLES
	////////////////////////////////
	omp_set_num_threads(GPUSTREAMS);
	/////////////////////////////////
	//END SET OPENMP ENVIRONMENT VARIABLES
	////////////////////////////////
	
	

	/////////////////////////////////
	//CREATE STREAMS
	////////////////////////////////

	cudaStream_t stream[GPUSTREAMS];
	
	for (int i=0; i<GPUSTREAMS; i++){
	cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
	}	

	/////////////////////////////////
	//END CREATE STREAMS
	////////////////////////////////
	
	

	///////////////////////////////////
	//LAUNCH KERNEL IN BATCHES
	///////////////////////////////////
		
	//since we use the strided scheme, some of the batch sizes
	//are off by 1 of each other, a first group of batches will
	//have 1 extra data point to process, and we calculate which batch numbers will 
	//have that.  The batchSize is the lower value (+1 is added to the first ones)

	CTYPE* dev_workCounts;
	cudaMalloc((void **)&dev_workCounts, sizeof(CTYPE)*2);
#if COUNTMETRICS == 1
		gpuErrchk(cudaMemcpy(dev_workCounts, workCounts, 2*sizeof(CTYPE), cudaMemcpyHostToDevice ));
#endif

	// unsigned int batchSize=(*DBSIZE)/numBatches;
	// unsigned int batchesThatHaveOneMore=(*DBSIZE)-(batchSize*numBatches); //batch number 0- < this value have one more
	// printf("\nBatches that have one more GPU thread: %u batchSize(N): %u, \n",batchesThatHaveOneMore,batchSize);

	uint64_t totalResultsLoop=0;

	/*
	// initialize array, size =  numBatches * # of groups
	unsigned int * batchStartsEachGroup;
	unsigned int * batchEndsEachGroup;
	batchStartsEachGroup=(unsigned int*)malloc(indexGroups->size()*numBatches*sizeof(unsigned int));
	batchEndsEachGroup=(unsigned int*)malloc(indexGroups->size()*numBatches*sizeof(unsigned int));
	// loop through each group
	for(int i=0; i<indexGroups->size(); i++) {
		// get approximate batch size for each (rounded up)
		unsigned int batchSize = (((*indexGroups)[i].indexmax - (*indexGroups)[i].indexmin)+ numBatches - 1) / numBatches;
		unsigned int currIdx = (*indexGroups)[i].indexmin;

		// start at 0
		batchStartsEachGroup[(i*numBatches)] = currIdx;

		// loop through each numBatches
		for( int j=0; j<numBatches; j++) {
			// if not end
			if( j < numBatches-1 ) {
				currIdx += batchSize;
				batchEndsEachGroup[(i*numBatches)+j] = currIdx;
				batchStartsEachGroup[(i*numBatches)+j+1] = currIdx;
			}
			else {
				batchEndsEachGroup[(i*numBatches)+j] = (*indexGroups)[i].indexmax;
			}
		}
	}
	*/

	/*
	for(int i=0; i<indexGroups->size(); i++) {
		printf("\nBatch Size:%d", (((*indexGroups)[i].indexmax - (*indexGroups)[i].indexmin)+ numBatches - 1) / numBatches);
		printf("\n%d: %d, %d\n", (*indexGroups)[i].index, (*indexGroups)[i].indexmin, (*indexGroups)[i].indexmax);
		for( int j=0; j<numBatches; j++) {
			printf("\nstart:%d", batchStartsEachGroup[(i*numBatches)+j]);
			printf("\nend:%d", batchEndsEachGroup[(i*numBatches)+j]);
		}
		printf("\n");
	}
	*/

	unsigned int * indexGroupOffset; 
	indexGroupOffset=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	
	unsigned int * dev_indexGroupOffset; 
	
	//allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_indexGroupOffset, sizeof(unsigned int)*GPUSTREAMS));

	unsigned int * batchesThatHaveOneMoreForEachGroup;
	unsigned int * batchSizeForEachGroup;
	batchesThatHaveOneMoreForEachGroup=(unsigned int*)malloc(indexGroups->size()*sizeof(unsigned int));
	batchSizeForEachGroup=(unsigned int*)malloc(indexGroups->size()*sizeof(unsigned int));
	for(int i=0; i<indexGroups->size(); i++) {
		batchSizeForEachGroup[i] = ((*indexGroups)[i].indexmax - (*indexGroups)[i].indexmin) / numBatchesEachIndex[i];
		batchesThatHaveOneMoreForEachGroup[i]=((*indexGroups)[i].indexmax - (*indexGroups)[i].indexmin)-(batchSizeForEachGroup[i]*numBatchesEachIndex[i]);
		printf("\nbatchSizeForEachGroup %d: %d", i, batchSizeForEachGroup[i]);
		printf("\nbatchesThatHaveOneMoreForEachGroup %d: %d", i, batchesThatHaveOneMoreForEachGroup[i]);
	}

	/*
	// allocate where each batch will run for each group to GPU
	unsigned int * dev_batchStartsEachGroup;
	unsigned int * dev_batchEndsEachGroup;
	gpuErrchk(cudaMalloc((void**)&dev_batchStartsEachGroup, indexGroups->size()*numBatches *sizeof(unsigned int)));
	gpuErrchk(cudaMemcpy(dev_batchStartsEachGroup, batchStartsEachGroup, indexGroups->size()*numBatches *sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**)&dev_batchEndsEachGroup, indexGroups->size()*numBatches *sizeof(unsigned int)));
	gpuErrchk(cudaMemcpy(dev_batchEndsEachGroup, batchEndsEachGroup, indexGroups->size()*numBatches *sizeof(unsigned int), cudaMemcpyHostToDevice));
	*/

	double totalKernelTime = 0;
	double totalTableConstructTime = 0;

	// count number of kernel invocations
	unsigned int numKernelInvocations;

		
		//FOR LOOP OVER THE NUMBER OF BATCHES STARTS HERE
		//i=0...numBatches
		#pragma omp parallel for schedule(static,1) reduction(+:totalResultsLoop) num_threads(GPUSTREAMS)
		for (int i=0; i<largestNumBatches; i++)
		// for (int i=0; i<1; i++)
		{	
			

			int tid=omp_get_thread_num();
			/*
			printf("\ntid: %d, starting iteration: %d",tid,i);
			//N NOW BECOMES THE NUMBER OF POINTS TO PROCESS PER BATCH
			//AS ONE GPU THREAD PROCESSES A SINGLE POINT

			if (i<batchesThatHaveOneMore)
			{
				N[tid]=batchSize+1;	
				printf("\nN (GPU threads): %d, tid: %d",N[tid], tid);
			}
			else
			{
				N[tid]=batchSize;	
				printf("\nN (1 less): %d tid: %d",N[tid], tid);
			}

			//set relevant parameters for the batched execution that get reset
			
			//copy N to device 
			//N IS THE NUMBER OF THREADS
			gpuErrchk(cudaMemcpyAsync( &dev_N[tid], &N[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] ));

			//the batched result set size (reset to 0):
			cnt[tid]=0;
			gpuErrchk(cudaMemcpyAsync( &dev_cnt[tid], &cnt[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] ));

			//the offset for batching, which keeps track of where to start processing at each batch
			batchOffset[tid]=numBatches; //for the strided
			gpuErrchk(cudaMemcpyAsync( &dev_offset[tid], &batchOffset[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] ));

			//the batch number for batching with strided
			batchNumber[tid]=i;
			gpuErrchk(cudaMemcpyAsync( &dev_batchNumber[tid], &batchNumber[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] ));


			const int TOTALBLOCKS=ceil((1.0*(N[tid]))/(1.0*BLOCKSIZE));	
			printf("\ntotal blocks: %d",TOTALBLOCKS);

			// note: blocksize is number of points running on a unit of the GPU
			//		total blocks is number of threads needed
			*/
			unsigned int gridIncrement =  0;
			for(int indexGroup=0; indexGroup<indexGroups->size(); indexGroup++) {
				// get index
				unsigned int whichIndex = (*indexGroups)[indexGroup].index;
				unsigned int whichDatabase = whichIndex % NUMRANDROTATIONS;

				if( i < numBatchesEachIndex[indexGroup] ) {
					
					printf("\ntid: %d, starting iteration: %d, Launch kernel for index offset %d",tid,i,whichIndex);

					if (tid >= GPUSTREAMS) {
							printf("ERROR: tid %d is out of bounds. It should be less than %d.\n", tid, GPUSTREAMS);
					}

					//the batched result set size (reset to 0):
					cnt[tid]=0;
					gpuErrchk(cudaMemcpyAsync( &dev_cnt[tid], &cnt[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] ));

					//N[tid]=batchEndsEachGroup[(indexGroup*numBatches)+i] - batchStartsEachGroup[(indexGroup*numBatches)+i];	
					//printf("\nN: %d tid: %d, batchStart: %d, batchEnd: %d", N[tid], tid, batchStartsEachGroup[(indexGroup*numBatches)+i], batchEndsEachGroup[(indexGroup*numBatches)+i]);
					if (i<batchesThatHaveOneMoreForEachGroup[indexGroup])
					{
						N[tid]=batchSizeForEachGroup[indexGroup]+1;	
						printf("\nN (GPU threads): %d, tid: %d",N[tid], tid);
					}
					else
					{
						N[tid]=batchSizeForEachGroup[indexGroup];	
						printf("\nN (1 less): %d tid: %d",N[tid], tid);
					}

					//copy N to device 
					//N IS THE NUMBER OF THREADS
					gpuErrchk(cudaMemcpyAsync( &dev_N[tid], &N[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] ));


					// for striding
					// batchOffset[tid]=batchStartsEachGroup[(indexGroup*numBatches)+i];
					batchOffset[tid]=numBatchesEachIndex[indexGroup];
					gpuErrchk(cudaMemcpyAsync( &dev_offset[tid], &batchOffset[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] ));

					// the offset to the index group and batch start
					indexGroupOffset[tid] = (*indexGroups)[indexGroup].indexmin + i;
					gpuErrchk(cudaMemcpyAsync( &dev_indexGroupOffset[tid], &indexGroupOffset[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] ));


					const int TOTALBLOCKS=ceil((1.0*(N[tid]))/(1.0*BLOCKSIZE));

					double kernelstart=omp_get_wtime();

					//execute kernel	
					//0 is shared memory pool
					kernelNDGridIndexGlobal<<< TOTALBLOCKS, BLOCKSIZE, 0, stream[tid]>>>(dev_debug1, dev_debug2, &dev_N[tid], 
						&dev_offset[tid], &dev_indexGroupOffset[tid], dev_database+(whichDatabase * (*DBSIZE) * GPUNUMDIM), dev_epsilon, dev_allGrids+gridIncrement, dev_allIndexLookupArr+(whichIndex * (*DBSIZE)), 
						dev_allGridCellLookupArr+gridIncrement, dev_allGridCellLookupArr+gridIncrement+allNNonEmptyCells[whichIndex], dev_allMinArr+(whichIndex * NUMINDEXEDDIM), 
						dev_allNCells+(whichIndex * NUMINDEXEDDIM), dev_orderedIndexPntIDs, &dev_cnt[tid], dev_pointIDKey[tid], dev_pointInDistValue[tid], 
						dev_workCounts);

					numKernelInvocations += 1;

					errCode=cudaDeviceSynchronize();
					cout <<"\n\nError from device synchronize: "<<errCode;

					double kernelend=omp_get_wtime();

					totalKernelTime += (kernelend - kernelstart);
					printf("\nKernel Invocation Time: %f",(kernelend - kernelstart));



					cout <<"\n\nKERNEL LAUNCH RETURN: "<<cudaGetLastError()<<endl<<endl;
					if ( cudaSuccess != cudaGetLastError() ){
						cout <<"\n\nERROR IN KERNEL LAUNCH. ERROR: "<<cudaSuccess<<endl<<endl;
					}
				
					// find the size of the number of results
					

					errCode=cudaMemcpyAsync( &cnt[tid], &dev_cnt[tid], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid] );
					if(errCode != cudaSuccess) {
					cout << "\nError: getting cnt from GPU Got error with code " << errCode << endl; 
					}
					else{
						// printf("\nGPU: result set size within epsilon (GPU grid): %d",cnt[tid]);
						fprintf(stderr,"\nGPU: result set size within epsilon (GPU grid): %d",cnt[tid]);
					}

					//add the batched result set size to the total count
					totalResultsLoop+=cnt[tid];

					////////////////////////////////////
					//SORT THE TABLE DATA ON THE GPU
					//THERE IS NO ORDERING BETWEEN EACH POINT AND THE ONES THAT IT'S WITHIN THE DISTANCE OF
					////////////////////////////////////

					//sort by key with the data already on the device:
					//wrap raw pointer with a device_ptr to use with Thrust functions
					thrust::device_ptr<int> dev_keys_ptr(dev_pointIDKey[tid]);
					thrust::device_ptr<int> dev_data_ptr(dev_pointInDistValue[tid]);

					//XXXXXXXXXXXXXXXX
					//THRUST USING STREAMS REQUIRES THRUST V1.8 
					//XXXXXXXXXXXXXXXX
					
					
					try{
					thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), dev_keys_ptr, dev_keys_ptr + cnt[tid], dev_data_ptr);


					}
					catch(std::bad_alloc &e)
					{
						std::cerr << "Ran out of memory while sorting, " << GPUBufferSize << std::endl;
						exit(-1);
					}
					


					//thrust with streams into individual buffers for each batch
					
					cudaMemcpyAsync(thrust::raw_pointer_cast(pointIDKey[tid]), thrust::raw_pointer_cast(dev_keys_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);
					cudaMemcpyAsync(thrust::raw_pointer_cast(pointInDistValue[tid]), thrust::raw_pointer_cast(dev_data_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);	

					//need to make sure the data is copied before constructing portion of the neighbor table
					cudaStreamSynchronize(stream[tid]);

					double tableconstuctstart=omp_get_wtime();
					//set the number of neighbors in the pointer struct:
					(*pointersToNeighbors)[i].sizeOfDataArr=cnt[tid];    
					(*pointersToNeighbors)[i].dataPtr=new int[cnt[tid]]; 

					////////////////////////////
					//New with multiple pointers to data arrays
					unsigned int uniqueCnt=0;
					unsigned int * dev_uniqueCnt; 
					
					//allocate on the device
					gpuErrchk(cudaMalloc((void**)&dev_uniqueCnt, sizeof(unsigned int)));

					//iniitalize the count to 0
					gpuErrchk(cudaMemcpyAsync( dev_uniqueCnt, &uniqueCnt, sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] ));

					//host side result
					int * uniqueKey=new int[cnt[tid]];
					int * uniqueKeyPosition=new int[cnt[tid]];

					int * dev_uniqueKey;
					int * dev_uniqueKeyPosition;

					
					
					//allocate memory on device:
					
					gpuErrchk(cudaMalloc( (void**)&dev_uniqueKey, sizeof(int)*(cnt[tid])));

					
					gpuErrchk(cudaMalloc( (void**)&dev_uniqueKeyPosition, sizeof(int)*(cnt[tid])));
					
			
					const int TOTALBLOCKS2=ceil((1.0*(cnt[tid]))/(1.0*BLOCKSIZE));	
					printf("\ntotal blocks: %d",TOTALBLOCKS2);

					//execute kernel for uniquing the keys	
					//0 is shared memory pool
					kernelUniqueKeys<<< TOTALBLOCKS2, BLOCKSIZE, 0, stream[tid]>>>(dev_pointIDKey[tid], &dev_cnt[tid], dev_uniqueKey, dev_uniqueKeyPosition, dev_uniqueCnt);

					cudaStreamSynchronize(stream[tid]);
					
					cout <<"\n\n UNIQUE KEY KERNEL LAUNCH RETURN: "<<cudaGetLastError()<<endl<<endl;
					if ( cudaSuccess != cudaGetLastError() ){
						cout <<"\n\nERROR IN UNIQUE KEY KERNEL LAUNCH. ERROR: "<<cudaSuccess<<endl<<endl;
					}
					//get the number of unique keys
					
					gpuErrchk(cudaMemcpyAsync( &uniqueCnt, dev_uniqueCnt, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid] ));
					cudaStreamSynchronize(stream[tid]);
					printf("\nGPU: unique keys (batch: %d): %u",i,uniqueCnt);fflush(stdout);
					

					

					


					//sort by key with the data already on the device:
					//wrap raw pointer with a device_ptr to use with Thrust functions
					thrust::device_ptr<int> dev_uniqueKey_ptr(dev_uniqueKey);
					thrust::device_ptr<int> dev_uniqueKeyPosition_ptr(dev_uniqueKeyPosition);

					try{
					thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), dev_uniqueKey_ptr, dev_uniqueKey_ptr + uniqueCnt, dev_uniqueKeyPosition_ptr);
					}
					catch(std::bad_alloc &e)
					{
						std::cerr << "Ran out of memory while sorting, " << GPUBufferSize << std::endl;
						exit(-1);
					}



					//thrust with streams into individual buffers for each batch
					cudaMemcpyAsync(thrust::raw_pointer_cast(uniqueKey), thrust::raw_pointer_cast(dev_uniqueKey_ptr), uniqueCnt*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);
					cudaMemcpyAsync(thrust::raw_pointer_cast(uniqueKeyPosition), thrust::raw_pointer_cast(dev_uniqueKeyPosition_ptr), uniqueCnt*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);	

					//need to make sure the data is copied before constructing portion of the neighbor table
					cudaStreamSynchronize(stream[tid]);



					constructNeighborTableKeyValueWithPtrsWithMultipleUpdatesMultipleDataArrays(pointIDKey[tid], pointInDistValue[tid], neighborTable, (*pointersToNeighbors)[i].dataPtr, &cnt[tid], uniqueKey, uniqueKeyPosition, uniqueCnt);

					cudaFree(dev_uniqueCnt);
					cudaFree(dev_uniqueKey);
					cudaFree(dev_uniqueKeyPosition);
					cudaStreamSynchronize(stream[tid]);
					
					double tableconstuctend=omp_get_wtime();	
					
					totalTableConstructTime += (tableconstuctend - tableconstuctstart);
					printf("\nTable construct time: %f", tableconstuctend - tableconstuctstart);


					printf("\nRunning total of total size of result array, tid: %d: %lu", tid, totalResultsLoop);
				}

				// increment grid for next random offset
				gridIncrement += allNNonEmptyCells[whichIndex];

			}		

		} //END LOOP OVER THE GPU BATCHES

	printf("\n");

#if COUNTMETRICS == 1
        cudaMemcpy(workCounts, dev_workCounts, 2*sizeof(CTYPE), cudaMemcpyDeviceToHost );
        printf("\nPoint comparisons: %llu, Cell evaluations: %llu", workCounts[0],workCounts[1]);
#endif

	printf("\nTotal number of kernel invocations: %d", numKernelInvocations);

	// printf("\nTotal Kernel Invocation Time: %f", totalKernelTime);
	// printf("\nTable construct time: %f", totalTableConstructTime);

	
	
	printf("\nTOTAL RESULT SET SIZE ON HOST:  %lu", totalResultsLoop);
	*totalNeighbors=totalResultsLoop;


	double tKernelResultsEnd=omp_get_wtime();
	
	printf("\nTime to launch kernel and execute everything (get results etc.) except freeing memory: %f",tKernelResultsEnd-tKernelResultsStart);




	///////////////////////////////////
	//END GET RESULT SET
	///////////////////////////////////



	///////////////////////////////////	
	//OPTIONAL DEBUG VALUES
	///////////////////////////////////
	
	// gpuErrchk(cudaMemcpy(debug1, dev_debug1, sizeof(unsigned int), cudaMemcpyDeviceToHost )); printf("\nDebug1 value: %u",*debug1);
	// gpuErrchk(cudaMemcpy(debug2, dev_debug2, sizeof(unsigned int), cudaMemcpyDeviceToHost ));printf("\nDebug2 value: %u",*debug2);	

	///////////////////////////////////	
	//END OPTIONAL DEBUG VALUES
	///////////////////////////////////
	

	///////////////////////////////////
	//FREE MEMORY FROM THE GPU
	///////////////////////////////////

	double tFreeStart=omp_get_wtime();

	for (int i=0; i<GPUSTREAMS; i++){
		errCode=cudaStreamDestroy(stream[i]);
		if(errCode != cudaSuccess) {
		cout << "\nError: destroying stream" << errCode << endl; 
		}
	}

	/*
	#if QUERYREORDER==1
	cudaFree(dev_orderedQueryPntIDs);
	#endif
	*/

	//free the data on the device
	cudaFree(dev_database);
	cudaFree(dev_debug1);
	cudaFree(dev_debug2);
	cudaFree(dev_allGrids);
	cudaFree(dev_allGridCellLookupArr);
	cudaFree(dev_allIndexLookupArr);
	cudaFree(dev_allMinArr);
	cudaFree(dev_allNCells);
	cudaFree(dev_allNNonEmptyCells);
	cudaFree(dev_N); 	
	cudaFree(dev_cnt); 
	cudaFree(dev_offset); 
	// cudaFree(dev_batchNumber); 
	cudaFree(dev_indexGroupOffset);

	free(database);
	free(totalResultSetCnt);
	free(cnt);
	free(numBatchesEachIndex);
	free(N);
	free(batchOffset);
	free(debug1);
	free(debug2);
	free(indexGroupOffset);
	free(batchesThatHaveOneMoreForEachGroup);
	free(batchSizeForEachGroup);

	
	//free data related to the individual streams for each batch
	for (int i=0; i<GPUSTREAMS; i++){
		//free the data on the device
		cudaFree(dev_pointIDKey[i]);
		cudaFree(dev_pointInDistValue[i]);

		//free on the host
		cudaFreeHost(pointIDKey[i]);
		cudaFreeHost(pointInDistValue[i]);
	}


	double tFreeEnd=omp_get_wtime();

	printf("\nTime freeing memory: %f", tFreeEnd - tFreeStart);
	// }
	cout<<"\n** last error at end of fn batches (could be from freeing memory): "<<cudaGetLastError();

	printf("\nTime to estimate batches: %f",tendbatchest - tstartbatchest);

	return (tKernelResultsEnd-tKernelResultsStart) - (tendbatchest - tstartbatchest);

}




void constructNeighborTableKeyValueWithPtrs(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt)
{
	#if STAMP==0
	
	//copy the value data:
	std::copy(pointInDistValue, pointInDistValue+(*cnt), pointersToNeighbors);



	//Step 1: find all of the unique keys and their positions in the key array
	unsigned int numUniqueKeys=0;

	std::vector<keyData> uniqueKeyData;

	keyData tmp;
	tmp.key=pointIDKey[0];
	tmp.position=0;
	uniqueKeyData.push_back(tmp);

	//we assign the ith data item when iterating over i+1th data item,
	//so we go 1 loop iteration beyond the number (*cnt)
	for (int i=1; i<(*cnt)+1; i++){
		if (pointIDKey[i-1]!=pointIDKey[i]){
			numUniqueKeys++;
			tmp.key=pointIDKey[i];
			tmp.position=i;
			uniqueKeyData.push_back(tmp);
		}
	}

	
	//insert into the neighbor table the values based on the positions of 
	//the unique keys obtained above. 
	for (int i=0; i<uniqueKeyData.size()-1; i++) {
		int keyElem=uniqueKeyData[i].key;
		neighborTable[keyElem].pointID=keyElem;
		neighborTable[keyElem].indexmin=uniqueKeyData[i].position;
		neighborTable[keyElem].indexmax=uniqueKeyData[i+1].position-1;
	
		//update the pointer to the data array for the values
		neighborTable[keyElem].dataPtr=pointersToNeighbors;	
	}
	#endif

	}




//Fixing the issue with unicomp requiring multiple updates and overwriting the data
void constructNeighborTableKeyValueWithPtrsWithMultipleUpdatesMultipleDataArrays(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt, int * uniqueKeys, int * uniqueKeyPosition, unsigned int numUniqueKeys)
{

	omp_set_max_active_levels(3);

	#pragma omp parallel for num_threads(8)
	for (unsigned int i=0; i<(*cnt); i++)
	{
		pointersToNeighbors[i]=pointInDistValue[i];
	}

	double tstartconstruct=omp_get_wtime();
	
	//if using unicomp we need to update different parts of the struct
	#if STAMP==1
	#pragma omp parallel for num_threads(8)
	for (unsigned int i=0; i<numUniqueKeys; i++) {

		int keyElem=uniqueKeys[i];
		//Update counter to write position in critical section
		pthread_mutex_lock(&neighborTable[keyElem].pointLock);
		
		int nextIdx=neighborTable[keyElem].cntNDataArrays;
		neighborTable[keyElem].cntNDataArrays++;
		pthread_mutex_unlock(&neighborTable[keyElem].pointLock);

		neighborTable[keyElem].vectindexmin[nextIdx]=uniqueKeyPosition[i];
		neighborTable[keyElem].vectdataPtr[nextIdx]=pointersToNeighbors;	

		//final value will be missing
		if (i==(numUniqueKeys-1))
		{
			neighborTable[keyElem].vectindexmax[nextIdx]=(*cnt)-1;
		}
		else
		{
			neighborTable[keyElem].vectindexmax[nextIdx]=(uniqueKeyPosition[i+1])-1;
		}
	}
	#endif


	//if not using unicomp we need to update the original parts of the struct
	#if STAMP==0
	#pragma omp parallel for num_threads(8)
	for (unsigned int i=0; i<numUniqueKeys; i++) {

			int keyElem=uniqueKeys[i];
			neighborTable[keyElem].pointID=keyElem;

			neighborTable[keyElem].indexmin=uniqueKeyPosition[i];
			neighborTable[keyElem].dataPtr=pointersToNeighbors;	

			//final value will be missing
			if (i==(numUniqueKeys-1))
			{
				neighborTable[keyElem].indexmax=(*cnt)-1;
			}
			else
			{
				neighborTable[keyElem].indexmax=(uniqueKeyPosition[i+1])-1;
			}
		}
	#endif

	double tendconstruct=omp_get_wtime();
	printf("\nTime to do the copy: %f", tendconstruct - tstartconstruct);
	
	return;


} //end function





void constructNeighborTableKeyValue(int * pointIDKey, int * pointInDistValue, struct table * neighborTable, unsigned int * cnt)
{
	
	//newer multithreaded way:
	//Step 1: find all of the unique keys and their positions in the key array
	
	//double tstart=omp_get_wtime();

	unsigned int numUniqueKeys=0;
	unsigned int count=0;

	

	std::vector<keyData> uniqueKeyData;

	keyData tmp;
	tmp.key=pointIDKey[0];
	tmp.position=0;
	uniqueKeyData.push_back(tmp);



	//we assign the ith data item when iterating over i+1th data item,
	//so we go 1 loop iteration beyond the number (*cnt)
	for (int i=1; i<(*cnt)+1; i++)
	{
		if (pointIDKey[i-1]!=pointIDKey[i])
		{
			numUniqueKeys++;
			tmp.key=pointIDKey[i];
			tmp.position=i;
			uniqueKeyData.push_back(tmp);
		}
	}



	//Step 2: In parallel, insert into the neighbor table the values based on the positions of 
	//the unique keys obtained above. Since multiple threads access this function, we don't want to 
	//do too many memory operations while GPU memory transfers are occurring, or else we decrease the speed that we 
	//get data off of the GPU
	omp_set_max_active_levels(3);
	#pragma omp parallel for reduction(+:count) num_threads(2) schedule(static,1)
	for (int i=0; i<uniqueKeyData.size()-1; i++) 
	{
		int keyElem=uniqueKeyData[i].key;
		int valStart=uniqueKeyData[i].position;
		int valEnd=uniqueKeyData[i+1].position-1;
		int size=valEnd-valStart+1;
		
		//seg fault from here: is it neighbortable mem alloc?
		neighborTable[keyElem].pointID=keyElem;
		neighborTable[keyElem].neighbors.insert(neighborTable[keyElem].neighbors.begin(),&pointInDistValue[valStart],&pointInDistValue[valStart+size]);
		count+=size;

	}
	

}





//Uses a brute force kernel to calculate the direct neighbors of the points in the database
void makeDistanceTableGPUBruteForce(std::vector<std::vector <DTYPE> > * NDdataPoints, DTYPE* epsilon, struct table * neighborTable, unsigned long long int * totalNeighbors)
{

	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////
	unsigned int * N;
	N=(unsigned int*)malloc(sizeof(unsigned int));
	*N=NDdataPoints->size();
	
	printf("\nIn main GPU method: Number of data points, (N), is: %u ",*N);cout.flush();



	
	//the database will just be a 1-D array, we access elemenets based on NDIM
	DTYPE* database= (DTYPE*)malloc(sizeof(DTYPE)*(*N)*GPUNUMDIM);  
	DTYPE* dev_database= (DTYPE*)malloc(sizeof(DTYPE)*(*N)*GPUNUMDIM);  
	

	//allocate memory on device:
	gpuErrchk(cudaMalloc( (void**)&dev_database, sizeof(DTYPE)*GPUNUMDIM*(*N)));
	//copy the database from the ND vector to the array:
	for (int i=0; i<*N; i++){
		std::copy((*NDdataPoints)[i].begin(), (*NDdataPoints)[i].end(), database+(i*GPUNUMDIM));
	}
	
	//copy database to the device:
	gpuErrchk(cudaMemcpy(dev_database, database, sizeof(DTYPE)*(*N)*GPUNUMDIM, cudaMemcpyHostToDevice));

	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////

	


	///////////////////////////////////
	//ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////
	//NON-PINNED MEMORY FOR SINGLE KERNEL INVOCATION (NO BATCHING)


	//CHANGING THE RESULTS TO KEY VALUE PAIR SORT, WHICH IS TWO ARRAYS
	//KEY IS THE POINT ID
	//THE VALUE IS THE POINT ID WITHIN THE DISTANCE OF KEY

	int * dev_pointIDKey; //key
	int * dev_pointInDistValue; //value

	//num elements for the result set
	#define BUFFERELEM 300000000 


	gpuErrchk(cudaMalloc((void **)&dev_pointIDKey, sizeof(int)*BUFFERELEM));
	gpuErrchk(cudaMalloc((void **)&dev_pointInDistValue, sizeof(int)*BUFFERELEM));
	printf("\nmemory requested for results (GiB): %f",(double)(sizeof(int)*2*BUFFERELEM)/(1024*1024*1024));

	///////////////////////////////////
	//END ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////


	///////////////////////////////////
	//SET OTHER KERNEL PARAMETERS
	///////////////////////////////////

	
	
	//count values
	unsigned long long int * cnt;
	cnt=(unsigned long long int*)malloc(sizeof(unsigned long long int));
	*cnt=0;

	unsigned long long int * dev_cnt; 
	dev_cnt=(unsigned long long int*)malloc(sizeof(unsigned long long int));
	*dev_cnt=0;

	//allocate on the device
	gpuErrchk(cudaMalloc((unsigned long long int**)&dev_cnt, sizeof(unsigned long long int)));
	
	gpuErrchk(cudaMemcpy( dev_cnt, cnt, sizeof(unsigned long long int), cudaMemcpyHostToDevice ));
	
	//Epsilon
	DTYPE* dev_epsilon;
	dev_epsilon=(DTYPE*)malloc(sizeof( DTYPE));

	//Allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_epsilon, sizeof(DTYPE)));
		
	//size of the database:
	unsigned int * dev_N; 
	dev_N=(unsigned int*)malloc(sizeof( unsigned int ));

	//allocate on the device
	gpuErrchk(cudaMalloc((void**)&dev_N, sizeof(unsigned int)));

	//debug values
	unsigned int * dev_debug1; 
	unsigned int * dev_debug2; 
	dev_debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug1=0;
	dev_debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug2=0;




	//allocate on the device
	gpuErrchk(cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int ) ));
	
	gpuErrchk(cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int ) ));
	
	//copy N, epsilon to the device
	//epsilon
	gpuErrchk(cudaMemcpy( dev_epsilon, epsilon, sizeof(DTYPE), cudaMemcpyHostToDevice ));
	
	//N (DATASET SIZE)
	gpuErrchk(cudaMemcpy( dev_N, N, sizeof(unsigned int), cudaMemcpyHostToDevice ));

	///////////////////////////////////
	//END SET OTHER KERNEL PARAMETERS
	///////////////////////////////////


	


	///////////////////////////////////
	//LAUNCH KERNEL
	///////////////////////////////////

	const int TOTALBLOCKS=ceil((1.0*(*N))/(1.0*BLOCKSIZE));	
	printf("\ntotal blocks: %d",TOTALBLOCKS);


	//execute kernel	

	
	double tkernel_start=omp_get_wtime();
	kernelBruteForce<<< TOTALBLOCKS, BLOCKSIZE >>>(dev_N, dev_debug1, dev_debug2, dev_epsilon, dev_cnt, dev_database, dev_pointIDKey, dev_pointInDistValue);
	if ( cudaSuccess != cudaGetLastError() ){
    	printf( "Error in kernel launch!\n" );
    }


    cudaDeviceSynchronize();
    double tkernel_end=omp_get_wtime();
    printf("\nTime for kernel only: %f", tkernel_end - tkernel_start);
    ///////////////////////////////////
	//END LAUNCH KERNEL
	///////////////////////////////////
    


    ///////////////////////////////////
	//GET RESULT SET
	///////////////////////////////////

	//first find the size of the number of results
	gpuErrchk(cudaMemcpy( cnt, dev_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost ));
	printf("\nGPU: result set size on within epsilon: %llu",*cnt);
	

	*totalNeighbors=(*cnt);

	//get debug information (optional)
	unsigned int * debug1;
	debug1=(unsigned int*)malloc(sizeof(unsigned int));
	*debug1=0;
	unsigned int * debug2;
	debug2=(unsigned int*)malloc(sizeof(unsigned int));
	*debug2=0;

	gpuErrchk(cudaMemcpy(debug1, dev_debug1, sizeof(unsigned int), cudaMemcpyDeviceToHost ));printf("\nDebug1 value: %u",*debug1);
	

	gpuErrchk(cudaMemcpy(debug2, dev_debug2, sizeof(unsigned int), cudaMemcpyDeviceToHost ));
	printf("\nDebug2 value: %u",*debug2);
	
	///////////////////////////////////
	//END GET RESULT SET
	///////////////////////////////////


	///////////////////////////////////
	//FREE MEMORY FROM THE GPU
	///////////////////////////////////
    //free:
	cudaFree(dev_database);
	cudaFree(dev_debug1);
	cudaFree(dev_debug2);
	cudaFree(dev_cnt);
	cudaFree(dev_epsilon);

	////////////////////////////////////



}




//Order the work from most to least work
//Based on the points within each cell
void computeWorkDifficulty(unsigned int * outputOrderedQueryPntIDs, struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, unsigned int * indexLookupArr, struct grid * index)
{
	std::vector<workArray> totalWork; 

	//loop over each non-empty cell and find the points contained within
	//record the number of points in the cell
	for (int i=0; i<*nNonEmptyCells; i++)
	{
		unsigned int grid_cell_idx=gridCellLookupArr[i].idx;		

		unsigned int numPtsInCell=(index[grid_cell_idx].indexmax-index[grid_cell_idx].indexmin)+1;

			for (int j=index[i].indexmin; j<=index[i].indexmax;j++)
			{
				workArray tmp;
				tmp.queryPntID=indexLookupArr[j];
				tmp.pntsInCell=numPtsInCell;
				totalWork.push_back(tmp);
			}



	}

	//sort the array containing the total work:
	std::sort(totalWork.begin(), totalWork.end(), compareWorkArrayByNumPointsInCell);

	for (unsigned int i=0; i<totalWork.size(); i++)
	{
		outputOrderedQueryPntIDs[i]=totalWork[i].queryPntID;
	}

	return;	

}

void warmUpGPU(){
// initialize all ten integers of a device_vector to 1 
thrust::device_vector<int> D(10, 1); 
// set the first seven elements of a vector to 9 
thrust::fill(D.begin(), D.begin() + 7, 9); 
// initialize a host_vector with the first five elements of D 
thrust::host_vector<int> H(D.begin(), D.begin() + 5); 
// set the elements of H to 0, 1, 2, 3, ... 
thrust::sequence(H.begin(), H.end()); // copy all of H back to the beginning of D 
thrust::copy(H.begin(), H.end(), D.begin()); 
// print D 
for(int i = 0; i < D.size(); i++) 
std::cout << " D[" << i << "] = " << D[i]; 
return;
}

//New: index on the GPU
//output:
//struct gridCellLookup ** gridCellLookupArr
//struct grid ** index
//unsigned int * indexLookupArr
void populateNDGridIndexAndLookupArrayGPU(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE *epsilon, DTYPE* minArr,  uint64_t totalCells, unsigned int * nCells, struct gridCellLookup ** gridCellLookupArr, struct grid ** index, unsigned int * indexLookupArr, unsigned int *nNonEmptyCells, std::vector<std::vector<int>> *incrementorVects, std::vector<workArrayPnt> *totalPointsWork)
{

	printf("\nIndexing on the GPU");

	//CUDA error code:
	cudaError_t errCode;


	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////

	unsigned int * DBSIZE;
	DBSIZE=(unsigned int*)malloc(sizeof(unsigned int));
	*DBSIZE=NDdataPoints->size();
	
	printf("\nDBSIZE is: %u",*DBSIZE);cout.flush();

	unsigned int * dev_DBSIZE;

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_DBSIZE, sizeof(unsigned int));		
	if(errCode != cudaSuccess) {
	cout << "\nError: database N -- error with code " << errCode << endl; cout.flush(); 
	}


	//copy database size to the device
	errCode=cudaMemcpy(dev_DBSIZE, DBSIZE, sizeof(unsigned int), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database size Got error with code " << errCode << endl; 
	}
	
	DTYPE* database= (DTYPE*)malloc(sizeof(DTYPE)*(*DBSIZE)*(GPUNUMDIM));  
	
	DTYPE* dev_database;  
		
	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_database, sizeof(DTYPE)*(GPUNUMDIM)*(*DBSIZE));		
	if(errCode != cudaSuccess) {
	cout << "\nError: database alloc -- error with code " << errCode << endl; cout.flush(); 
	}


	//copy the database from the ND vector to the array:
	for (int i=0; i<(*DBSIZE); i++){
		std::copy((*NDdataPoints)[i].begin(), (*NDdataPoints)[i].end(), database+(i*(GPUNUMDIM)));
	}


	//copy database to the device
	errCode=cudaMemcpy(dev_database, database, sizeof(DTYPE)*(GPUNUMDIM)*(*DBSIZE), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database2 Got error with code " << errCode << endl; 
	}




	//printf("\n size of database: %d",N);



	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////


	///////////////////////////////////
	//COPY GRID DIMENSIONS TO THE GPU
	//THIS INCLUDES THE NUMBER OF CELLS IN EACH DIMENSION, 
	//AND THE STARTING POINT OF THE GRID IN THE DIMENSIONS 
	///////////////////////////////////

	//minimum boundary of the grid:
	DTYPE* dev_minArr;

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_minArr, sizeof(DTYPE)*(NUMINDEXEDDIM));
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc minArr -- error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_minArr, minArr, sizeof(DTYPE)*(NUMINDEXEDDIM), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: Copy minArr to device -- error with code " << errCode << endl; 
	}	


	//number of cells in each dimension
	unsigned int * dev_nCells;
	

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_nCells, sizeof(unsigned int)*(NUMINDEXEDDIM));
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc nCells -- error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_nCells, nCells, sizeof(unsigned int)*(NUMINDEXEDDIM), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: Copy nCells to device -- error with code " << errCode << endl; 
	}	


	///////////////////////////////////
	//END COPY GRID DIMENSIONS TO THE GPU
	///////////////////////////////////


	///////////////////////////////////
	//EPSILON
	///////////////////////////////////
	DTYPE* dev_epsilon;
	// dev_epsilon=(DTYPE*)malloc(sizeof( DTYPE));
	

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_epsilon, sizeof(DTYPE));
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc epsilon -- error with code " << errCode << endl; 
	}

	//copy to device
	errCode=cudaMemcpy( dev_epsilon, epsilon, sizeof(DTYPE), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon copy to device -- error with code " << errCode << endl; 
	}		

	///////////////////////////////////
	//END EPSILON
	///////////////////////////////////


	///////////////////////////////////
	//Array for each point's cell
	///////////////////////////////////
	uint64_t * dev_pointCellArr;  
		
	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_pointCellArr, sizeof(uint64_t)*(*DBSIZE));		
	if(errCode != cudaSuccess) {
	cout << "\nError: point cell array alloc -- error with code " << errCode << endl; cout.flush(); 
	}

	///////////////////////////////////
	//End array for each point's cell
	///////////////////////////////////


	//First: we get the number of non-empty grid cells

	const int TOTALBLOCKS=ceil((1.0*(*DBSIZE))/(1.0*BLOCKSIZE));	
	printf("\ntotal blocks: %d",TOTALBLOCKS);

	kernelIndexComputeNonemptyCells<<< TOTALBLOCKS, BLOCKSIZE>>>(dev_database, dev_DBSIZE, dev_epsilon, dev_minArr, dev_nCells, dev_pointCellArr);

	cudaDeviceSynchronize();

	// create unique cells array in device so that point cells do not have to be recomputed:
	uint64_t * dev_uniqueCellArr;

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_uniqueCellArr, sizeof(uint64_t)*(*DBSIZE));		
	if(errCode != cudaSuccess) {
	cout << "\nError: incrementors size -- error with code " << errCode << endl; cout.flush(); 
	}

	//copy incrementors size to the device
	errCode=cudaMemcpy(dev_uniqueCellArr, dev_pointCellArr, sizeof(uint64_t)*(*DBSIZE), cudaMemcpyDeviceToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: incrementors size Got error with code " << errCode << endl; 
	}

	thrust::device_ptr<uint64_t> dev_uniqueCellArr_ptr(dev_uniqueCellArr);

	thrust::device_ptr<uint64_t> dev_new_end;

	try{
		//first sort
		thrust::sort(thrust::device, dev_uniqueCellArr_ptr, dev_uniqueCellArr_ptr + (*DBSIZE)); //, thrust::greater<uint64_t>()
		//then unique
		dev_new_end=thrust::unique(thrust::device, dev_uniqueCellArr_ptr, dev_uniqueCellArr_ptr + (*DBSIZE));
	}
	catch(std::bad_alloc &e)
	{
	 	std::cerr << "Ran out of memory while sorting, "<< std::endl;
	    exit(-1);
    }



    uint64_t * new_end = thrust::raw_pointer_cast(dev_new_end);

    uint64_t numNonEmptyCells=std::distance(dev_uniqueCellArr_ptr,dev_new_end);

    printf("\nNumber of full cells (non-empty): %lu",numNonEmptyCells);

    *nNonEmptyCells=numNonEmptyCells;



    ////////////////////////
    //populate grid cell lookup array- its the uniqued array above (dev_uniqueCellArr)
    *gridCellLookupArr= new struct gridCellLookup[numNonEmptyCells];

    uint64_t * pointCellArrTmp;  
    pointCellArrTmp=(uint64_t *)malloc((sizeof(uint64_t)*numNonEmptyCells));


    errCode=cudaMemcpy(pointCellArrTmp, dev_uniqueCellArr, sizeof(uint64_t)*numNonEmptyCells, cudaMemcpyDeviceToHost);
	if(errCode != cudaSuccess) {
	cout << "\nError: pointCellArrTmp memcpy Got error with code " << errCode << endl; 
	}

	for (uint64_t i=0; i<numNonEmptyCells; i++)
	{
		(*gridCellLookupArr)[i].idx=i;
		(*gridCellLookupArr)[i].gridLinearID=pointCellArrTmp[i];
	}

	

	// for (uint64_t i=0; i<numNonEmptyCells; i++)
	// {
	// 	printf("\nGrid idx: %lu, linearID: %lu",(*gridCellLookupArr)[i].idx,(*gridCellLookupArr)[i].gridLinearID);
	// }


	
    //Second: we execute the same kernel again to get each point's cell ID-- we ruined this array by uniquing it
    //So that we don't run out of memory on the GPU for larger datasets, we redo the kernel and sort		
    // const int TOTALBLOCKS=ceil((1.0*(*DBSIZE))/(1.0*BLOCKSIZE));	
	// printf("\ntotal blocks: %d",TOTALBLOCKS);

    //Compute again pointCellArr-- the first one was invalidated because of the unique
	// kernelIndexComputeNonemptyCells<<< TOTALBLOCKS, BLOCKSIZE>>>(dev_database, dev_DBSIZE, dev_epsilon, dev_minArr, dev_nCells, dev_pointCellArr);




	//Create "values" for key/value pairs, that are the point idx of the database
	unsigned int * dev_databaseVal;

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_databaseVal, sizeof(unsigned int)*(*DBSIZE));
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc databaseVal -- error with code " << errCode << endl; 
	}

	kernelInitEnumerateDB<<< TOTALBLOCKS, BLOCKSIZE>>>(dev_databaseVal, dev_DBSIZE);

	//Sort the point ids by cell ids, using key/value pairs, where 
	//key-cell id, value- point id

	try
	{
	thrust::sort_by_key(thrust::device, dev_pointCellArr, dev_pointCellArr+(*DBSIZE),dev_databaseVal);
	}
	catch(std::bad_alloc &e)
	{
		std::cerr << "Ran out of memory while sorting key/value pairs "<< std::endl;
	    exit(-1);	
	}

	uint64_t * cellKey=(uint64_t *)malloc(sizeof(uint64_t)*(*DBSIZE));
	// unsigned int * databaseIDValue=(unsigned int *)malloc(sizeof(unsigned int)*(*DBSIZE));

	//Sorted keys by cell, aligning with the database point IDs below
	//keys
	errCode=cudaMemcpy(cellKey, dev_pointCellArr, sizeof(uint64_t)*(*DBSIZE), cudaMemcpyDeviceToHost);
	if(errCode != cudaSuccess) {
	cout << "\nError: pointCellArr memcpy Got error with code " << errCode << endl; 
	}
	//Point ids
	// errCode=cudaMemcpy(databaseIDValue, dev_databaseVal, sizeof(unsigned int)*(*DBSIZE), cudaMemcpyDeviceToHost);
	// if(errCode != cudaSuccess) {
	// cout << "\nError: databaseIDValue memcpy Got error with code " << errCode << endl; 
	// }

	//Point ids
	//indexLookupArr
	errCode=cudaMemcpy(indexLookupArr, dev_databaseVal, sizeof(unsigned int)*(*DBSIZE), cudaMemcpyDeviceToHost);
	if(errCode != cudaSuccess) {
	cout << "\nError: databaseIDValue memcpy Got error with code " << errCode << endl; 
	}


	//populate lookup array from grid cells into database (databaseIDValue above)


	//indexLookupArr
	// std::copy(databaseIDValue, databaseIDValue+(*DBSIZE), indexLookupArr);




	//Populate grid index
	//allocate memory for the index that will be sent to the GPU
	*index=new grid[numNonEmptyCells];
	


	//populate grid index

	//the first index
	(*index)[0].indexmin=0;

	uint64_t cnt=0;
	
	for (uint64_t i=1; i<(*DBSIZE); i++){
		
		if (cellKey[i-1]!=cellKey[i])
		{
			//grid index
			cnt++;
			(*index)[cnt].indexmin=i;			
			(*index)[cnt-1].indexmax=i-1;			

			
		}


		// (*index)[i].indexmin=tmpIndex[i].indexmin;
		// (*index)[i].indexmax=tmpIndex[i].indexmax;
		// (*gridCellLookupArr)[i].idx=i;
		// (*gridCellLookupArr)[i].gridLinearID=uniqueGridCellLinearIdsVect[i];
	}
	
	//the last index
	(*index)[numNonEmptyCells-1].indexmax=(*DBSIZE)-1;

	// for (int i=0; i<(*DBSIZE); i++)
	// {
	// 	printf("\npoint: %u, Cell: %llu",databaseIDValue[i],cellKey[i]);
	// }


	////////////////////////
	// get number distance calcs for each cell
	////////////////////////

	// allocate memory for number of non empty cells
	unsigned int* dev_nNonEmptyCells;
	errCode=cudaMalloc((void**)&dev_nNonEmptyCells, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc epsilon -- error with code " << errCode << endl; 
	}

	//copy to device
	errCode=cudaMemcpy( dev_nNonEmptyCells, nNonEmptyCells, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon copy to device -- error with code " << errCode << endl; 
	}

	
	//allocate memory on device:
	unsigned int * dev_nAdjCells;
	unsigned int nAdjCells = incrementorVects->size();

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_nAdjCells, sizeof(unsigned int));		
	if(errCode != cudaSuccess) {
	cout << "\nError: incrementors size -- error with code " << errCode << endl; cout.flush(); 
	}

	//copy incrementors size to the device
	errCode=cudaMemcpy(dev_nAdjCells, &(nAdjCells), sizeof(unsigned int), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: incrementors size Got error with code " << errCode << endl; 
	}


	// allocate memory for incremetor vectors
	int* incrementors = (int*)malloc(sizeof(int)*(nAdjCells)*(NUMINDEXEDDIM));  
	
	int* dev_incrementors;  

	errCode=cudaMalloc( (void**)&dev_incrementors, sizeof(int)*(nAdjCells)*(NUMINDEXEDDIM));		
	if(errCode != cudaSuccess) {
	cout << "\nError: incrementors alloc -- error with code " << errCode << endl; cout.flush(); 
	}

	//copy the incrementors from the incrementors vector to the array:
	for (int i=0; i<nAdjCells; i++){
		std::copy((*incrementorVects)[i].begin(), (*incrementorVects)[i].end(), incrementors+(i*(NUMINDEXEDDIM)));
	}

	//copy incrementors to the device
	errCode=cudaMemcpy(dev_incrementors, incrementors, sizeof(int)*(nAdjCells)*(NUMINDEXEDDIM), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: incrementors Got error with code " << errCode << endl; 
	}


	// allocate memory for adjacent cell matrix
	uint64_t * dev_cellDistCalcArr; 
		
	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_cellDistCalcArr, sizeof(uint64_t)*(*nNonEmptyCells));		
	if(errCode != cudaSuccess) {
	cout << "\nError: point cell array alloc -- error with code " << errCode << endl; cout.flush(); 
	}

		// get number of points in each cell
	uint64_t *host_cellNumPointsArr = (uint64_t*)malloc(sizeof(uint64_t) * numNonEmptyCells);
	for (int i=0; i<(numNonEmptyCells); i++)
	{
		host_cellNumPointsArr[i] = ((*index)[i].indexmax - (*index)[i].indexmin) + 1;

	}

	uint64_t * dev_cellNumPointsArr;

	// allocate memory in device
	errCode=cudaMalloc( (void**)&dev_cellNumPointsArr, sizeof(uint64_t) * numNonEmptyCells);		
	if(errCode != cudaSuccess) {
	cout << "\nError: cell num points array alloc -- error with code " << errCode << endl; cout.flush(); 
	}

	// copy array to device
	errCode=cudaMemcpy(dev_cellNumPointsArr, host_cellNumPointsArr,  sizeof(uint64_t) * numNonEmptyCells, cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: copying cell num points array to device Got error with code " << errCode << endl; 
	}

	
	const int TOTAL_NONEMPTY_CELL_BLOCKS=ceil((1.0*(numNonEmptyCells))/(1.0*BLOCKSIZE));
	kernelIndexComputeAdjacentCells<<< TOTAL_NONEMPTY_CELL_BLOCKS, BLOCKSIZE>>>(dev_cellDistCalcArr, dev_uniqueCellArr, dev_cellNumPointsArr, dev_nCells, dev_nNonEmptyCells, dev_incrementors, dev_nAdjCells);
	
	cudaDeviceSynchronize();

	////////////////////////
	// end number distance calcs for each cell
	////////////////////////

	////////////////////////
	// map number distance calcs for each point
	////////////////////////

	uint64_t * dev_pointDistCalcArr; 
		
	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_pointDistCalcArr, sizeof(uint64_t)*(*DBSIZE));		
	if(errCode != cudaSuccess) {
	cout << "\nError: point cell array alloc -- error with code " << errCode << endl; cout.flush(); 
	}

	kernelMapPointToNumDistCalcs<<< TOTALBLOCKS, BLOCKSIZE>>>(dev_pointDistCalcArr, dev_database, dev_DBSIZE, dev_epsilon, dev_minArr, dev_nCells, dev_cellDistCalcArr, dev_uniqueCellArr, dev_nNonEmptyCells);

	cudaDeviceSynchronize();

	// populate total work vector
	//allocate memory on host
	uint64_t * host_distCalcArr;
	host_distCalcArr = (uint64_t *)malloc(sizeof(uint64_t)*(*DBSIZE));

	// copy adj cell array from device to hsot
	cudaMemcpy(host_distCalcArr, dev_pointDistCalcArr, sizeof(uint64_t)*(*DBSIZE), cudaMemcpyDeviceToHost);

	cout << endl;
	for(int i=0; i<*DBSIZE; i++)
	{
		workArrayPnt tmp;
		tmp.pntIdx=i;
		tmp.numDistCalcs=host_distCalcArr[i];
		totalPointsWork->push_back(tmp);
	}
	
	
	////////////////////////
	// end map adjacent cells for each point
	////////////////////////




	printf("\nFull cells: %d (%f, fraction full)",(unsigned int)numNonEmptyCells, numNonEmptyCells*1.0/double(totalCells));
	printf("\nEmpty cells: %ld (%f, fraction empty)",totalCells-(unsigned int)numNonEmptyCells, (totalCells-numNonEmptyCells*1.0)/double(totalCells));
	printf("\nSize of index that would be sent to GPU (GiB) -- (if full index sent), excluding the data lookup arr: %f", (double)sizeof(struct grid)*(totalCells)/(1024.0*1024.0*1024.0));
	printf("\nSize of compressed index to be sent to GPU (GiB) , excluding the data and grid lookup arr: %f", (double)sizeof(struct grid)*(numNonEmptyCells*1.0)/(1024.0*1024.0*1024.0));
	printf("\nWhen copying from entire index to compressed index: number of non-empty cells: %lu",numNonEmptyCells);


	
	free(DBSIZE);
	free(database);
	free(pointCellArrTmp);
	free(cellKey);
	free(host_cellNumPointsArr);
	free(host_distCalcArr);
	

	// free(databaseIDValue);
	cudaFree(dev_DBSIZE);
	cudaFree(dev_database);
	cudaFree(dev_minArr);
	cudaFree(dev_nCells);
	cudaFree(dev_epsilon);
	cudaFree(dev_pointCellArr);
	cudaFree(dev_databaseVal);
	
	cudaFree(dev_nNonEmptyCells);
	cudaFree(dev_nAdjCells);
	cudaFree(dev_incrementors);
	cudaFree(dev_cellDistCalcArr);
	cudaFree(dev_pointDistCalcArr);
	cudaFree(dev_cellNumPointsArr);



}


void rotateOnGPU(DTYPE * dev_database, const unsigned int DBSIZE, const unsigned int whichDatabase) {
	// define random engine
    std::random_device rd;
	// initialize generator
    std::mt19937 gen(rd());
	// produce random float within [0, 2pi)
	std::uniform_real_distribution<> rad_dis(DTYPE(0), 2 * M_PI);
	// produce random int within [0, NUMINDEXEDDIM - 1]
	std::uniform_int_distribution<> index_dis(0, NUMINDEXEDDIM - 1);

	//CUDA error code:
	cudaError_t errCode;

	///////////////////////////////////
	//COPY DIMENSION PAIR TO GPU
	///////////////////////////////////
	unsigned int * dimPair = (unsigned int *)malloc(sizeof(unsigned int) * 2 * NUMPAIRROTATIONS);
	DTYPE * theta = (DTYPE *)malloc(sizeof(DTYPE) * NUMPAIRROTATIONS);

	for( unsigned int i=0; i<NUMPAIRROTATIONS; i++ )
	{
		// use first two dimensions for now
		dimPair[i * 2] = index_dis(gen);
		do {
			dimPair[i * 2 + 1] = index_dis(gen);
		} while( dimPair[i * 2 + 1] == dimPair[i * 2]);

		theta[i] = rad_dis(gen);

		printf("\nPair rotation %d:", i + 1);
		printf("\nPair: %d, %d / Theta: %f", dimPair[i * 2], dimPair[i * 2 + 1], theta[i]);	
	}

	unsigned int * dev_dimPair;
	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_dimPair, sizeof(unsigned int) * 2 * NUMPAIRROTATIONS);		
	if(errCode != cudaSuccess) {
	cout << "\nError: dimPair -- error with code " << errCode << endl; cout.flush(); 
	}

	//copy dimPair to the device
	errCode=cudaMemcpy(dev_dimPair, dimPair, sizeof(unsigned int) * 2 * NUMPAIRROTATIONS, cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: dimPair Got error with code " << errCode << endl; 
	}
	///////////////////////////////////
	//END COPY DIMENSION PAIR TO GPU
	///////////////////////////////////

	///////////////////////////////////
	//COPY THETA TO GPU
	///////////////////////////////////

	DTYPE * dev_theta;
	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_theta, sizeof(DTYPE) * NUMPAIRROTATIONS);		
	if(errCode != cudaSuccess) {
	cout << "\nError: theta -- error with code " << errCode << endl; cout.flush(); 
	}

	//copy theta to the device
	errCode=cudaMemcpy(dev_theta, theta, sizeof(DTYPE) * NUMPAIRROTATIONS, cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: theta Got error with code " << errCode << endl; 
	}
	///////////////////////////////////
	//END COPY THETA TO GPU
	///////////////////////////////////


	const int TOTALBLOCKS=ceil((1.0*(DBSIZE))/(1.0*BLOCKSIZE));

	kernelPairwiseDatabaseRotation <<< TOTALBLOCKS, BLOCKSIZE >>> (dev_database, DBSIZE, whichDatabase, dev_theta, dev_dimPair);

	printf("\n");

	cudaFree(dev_theta);
	cudaFree(dev_dimPair);
	free(dimPair);
	free(theta);
}