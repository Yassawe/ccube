#include "ccube.h"
#include <cuda.h>


// for now, allocate all ones
void allocateMemoryBuffers(struct Node* tree, int message_size){
    for(int i = 0; i<P; i++){
        cudaMalloc((void **)&tree[i].buffer, message_size*sizeof(float));
        cudaMemset(tree[i].buffer, 1, message_size*sizeof(float));
    }
}


void freeMemoryBuffers(struct Node* tree){
    for(int i = 0; i<P; i++){
        cudaFree(tree[i].buffer);
    }
}