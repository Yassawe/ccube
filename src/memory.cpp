#include "ccube.h"
#include <cuda.h>


// for now, allocate all ones
void allocateMemoryBuffers(struct Node* tree, int message_size){
    float* tmp = (float*)malloc(message_size*sizeof(float));

    for (int i = 0; i<message_size; i++){
        tmp[i] = 1;
    }

    for(int i = 0; i<P; i++){
        cudaSetDevice(i);
        cudaMalloc((void **)&tree[i].buffer, message_size*sizeof(float));
        cudaMemcpy(tree[i].buffer, tmp, message_size*sizeof(float), cudaMemcpyHostToDevice);
    }
    free(tmp);
}

void allocateLocks(struct Node* tree, int rank){
    int* tmp = (int *) malloc(NUM_BLOCKS*sizeof(int));
    
    for (int i =0; i < NUM_BLOCKS; i++){
        tmp[i] = 0;
    }
    
    cudaSetDevice(rank);

    cudaMalloc((void **)&tree[rank].r_lock, NUM_BLOCKS*sizeof(int));
    cudaMemcpy(tree[rank].r_lock, tmp, NUM_BLOCKS*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&tree[rank].b_lock, NUM_BLOCKS*sizeof(int));
    cudaMemcpy(tree[rank].b_lock, tmp, NUM_BLOCKS*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&tree[rank].r_ready, NUM_BLOCKS*sizeof(int));
    cudaMemcpy(tree[rank].r_ready, tmp, NUM_BLOCKS*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&tree[rank].b_ready, NUM_BLOCKS*sizeof(int));
    cudaMemcpy(tree[rank].b_ready, tmp, NUM_BLOCKS*sizeof(int), cudaMemcpyHostToDevice);
}

void freeMemoryBuffers(struct Node* tree){
    for(int i = 0; i<P; i++){
        cudaSetDevice(i);
        cudaFree(tree[i].buffer);
    }
}

void test(struct Node* tree, int rank, int target, int message_size){
    float* tmp = (float*)malloc(message_size*sizeof(float));

    cudaSetDevice(rank); 
    cudaError_t err = cudaMemcpy(tmp, tree[rank].buffer, message_size*sizeof(float), cudaMemcpyDeviceToHost);
    for (int j = 0; j<message_size; j++){
        if (tmp[j]!=target){
            printf("error at device %d, index %d. Value %.2f. Error %d\n", rank, j, tmp[j],err);
            return;
        }
    }
    
    printf("device %d, test passed\n", rank);
    free(tmp);
}
