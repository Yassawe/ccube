#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include "pthread.h"

#define P 3
#define CHUNK_SIZE 1024 //in float32 elements. hardcoded for now.
#define BLOCK_SIZE 1024
// for now i assume that chunk perfectly divides the message, for simplicity
// in-place operation is assumed for simplicity

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
    int num_chunks;
};

void createCommunicator(struct Node* tree);
void killCommunicator(struct Node* tree);
void allocateMemoryBuffers(struct Node* tree, int message_size);
void freeMemoryBuffers(struct Node* tree);

void test(struct Node* tree, int rank, int target, int message_size);

int launch(struct Node* tree, int rank, int parent, int child, int num_chunks);

void check_p2p();
void test_sum(struct Node* tree,int rank, int peer, int num_chunks);