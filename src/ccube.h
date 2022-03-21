#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include "pthread.h"

#define P 4
#define CHUNK_SIZE (1024*1024) //in float32 elements. hardcoded for now.
#define BLOCK_SIZE 1024
#define NUM_BLOCKS ((CHUNK_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE)



struct Node {
    cudaStream_t stream; 

    int parent;
    int child;
    
    int* lock;
    int* ready;

    float *buffer; 
};

struct t_args{
    struct Node * tree;
    int rank;
    int message_size;
};

void createCommunicator(struct Node* tree);
void killCommunicator(struct Node* tree);
void allocateMemoryBuffers(struct Node* tree, int message_size);
void allocateLocks(struct Node* tree, int rank);
void freeMemoryBuffers(struct Node* tree);
int launch(struct Node* tree, int rank, int message_size);

// [DEBUG]
void test(struct Node* tree, int rank, int target, int message_size);