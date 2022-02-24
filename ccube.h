#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

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
}

void createCommunicator(struct Node* tree);
void killCommunicator(struct Node* tree, int p);
void allreduce(struct Node* tree, int rank, int num_chunks);
