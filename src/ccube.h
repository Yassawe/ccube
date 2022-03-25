#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include "pthread.h"

#define P 8
#define CHUNK_SIZE (1048576) //in float32 elements. hardcoded for now.
#define BLOCK_SIZE 512
#define NUM_BLOCKS 4

// for now i assume that chunk perfectly divides the message, for simplicity
// in-place operation is assumed for simplicity

struct Node {
    cudaStream_t R_stream; 
    cudaStream_t B_stream;

    int parent;
    int left;
    int right;

    int* r_lock;
    int* b_lock;

    int* r_ready;
    int* b_ready;

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


void test(struct Node* tree, int rank, int target, int message_size);

