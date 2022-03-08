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
        cudaError_t err = cudaMemcpy(tmp, tree[i].buffer, message_size*sizeof(float), cudaMemcpyDeviceToHost);
        for (int j = 0; j<message_size; j++){
            if (tmp[j]!=P){
                printf("error at device %d, index %d. Value %.2f. Error %d\n", i, j, tmp[j],err);
                return;
            }
        }
    }
    printf("test passed\n");
    free(tmp);
}

