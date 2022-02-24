#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

struct Node {
    cudaStream_t R_stream; 
    cudaStream_t B_stream;

    int parent;
    int left;
    int right;

    int* r_lock;
    int* b_lock;

    int* r_done;
    int* b_done;

    float *buffer; 
}

void createCommunicator(struct Node* tree);
void killCommunicator(struct Node* tree, int p);
void allreduce(struct Node* tree, int rank, int num_chunks);
