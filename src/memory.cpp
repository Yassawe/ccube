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

        //printf("SET. DEVICE %d ADDRESS %p\n", i, (void *)tree[i].buffer);

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
    
    printf("test passed\n");
    free(tmp);
}

void check_p2p(){
    int check;

    cudaSetDevice(0);
    cudaDeviceCanAccessPeer(&check, 0, 2);
    printf("device 0->2 status: %d\n", check);

    cudaSetDevice(1);
    cudaDeviceCanAccessPeer(&check, 1, 2);
    printf("device 1->2 status: %d\n", check);

    cudaSetDevice(2);
    cudaDeviceCanAccessPeer(&check, 2, 0);
    printf("device 2->0 status: %d\n", check);
    cudaDeviceCanAccessPeer(&check, 2, 1);
    printf("device 2->1 status: %d\n", check);
    cudaDeviceCanAccessPeer(&check, 2, 3);
    printf("device 2->3 status: %d\n", check);

    cudaSetDevice(3);
    cudaDeviceCanAccessPeer(&check, 3, 2);
    printf("device 3->2 status: %d\n", check);
}