#include "ccube.h"
#include <cuda.h>


// for now, allocate all ones
void allocateMemoryBuffers(struct Node* tree, int message_size){
    for(int i = 0; i<P; i++){
        cudaSetDevice(i);
        cudaMalloc((void **)&tree[i].buffer, message_size*sizeof(float));
        cudaMemset(tree[i].buffer, 1, message_size*sizeof(float));
    }
}


void freeMemoryBuffers(struct Node* tree){
    for(int i = 0; i<P; i++){
        cudaSetDevice(i);
        cudaFree(tree[i].buffer);
    }
}

void test(struct Node* tree, int message_size){
    float* tmp = (float*)malloc(message_size*sizeof(float));

    for (int i =0; i<P; i++){
        cudaSetDevice(i);
        cudaMemcpy(tmp, tree[i].buffer, message_size*sizeof(float), cudaMemcpyDeviceToHost);
        
        //host implementation because i'm lazy
        for (int j=0; j<message_size; j++){
            if(tmp[j]!=P){
                printf("error in device %d, index %d. Value is %.2f", i, j, tmp[j]);
                return;
            }
        }
    }

    printf("test passed");
}