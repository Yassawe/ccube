#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define P 4
#define CHUNK_SIZE 2048 //in float32 elements. hardcoded for now.
#define BLOCK_SIZE 512


struct Node {
    cudaStream_t R_stream; 
    cudaStream_t B_stream;

    int parent;
    int left;
    int right;

    volatile int* r_lock;
    volatile int* b_lock;

    volatile int* r_done;
    volatile int* b_done;

    float *buffer; 
};

struct t_args{
    struct Node * tree;
    int rank;
    int num_chunks;
};

void createCommunicator(struct Node* tree);
void killCommunicator(struct Node* tree, int p);
void allreduce(struct t_args* args);
