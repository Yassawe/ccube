#include "ccube.h"
#include <cuda.h>


// for now, allocate all ones
void allocateMemoryBuffers(struct Node* tree, int message_size){
    for(int i = 0; i<P; i++){
        cudaSetDevice(i);
        cudaMalloc((void **)&tree[i].buffer, message_size*sizeof(float));
        cudaMemset(tree[i].buffer, 1, message_size*sizeof(float));
        printf("set. device %d. pointer %p\n", i, (void *)tree[i].buffer);
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
        //cudaError_t err = 
        cudaMemcpy(tmp, tree[i].buffer, message_size*sizeof(float), cudaMemcpyDeviceToHost);
        printf("check. device %d. pointer %p\n", i, (void *)tree[i].buffer);
        // for (int j=0; j<message_size; j++){
        //     if(tmp[j]!=P){
        //         printf("error in device %d, index %d. Value is %.2f. Error %d\n", i, j, tmp[j], err);
        //     }
        // }
    }

    //printf("test passed\n");
    free(tmp);
}

