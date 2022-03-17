#include "ccube.h"
#include <cuda.h>


// for now, allocate all ones
void allocateMemoryBuffers(struct Node* tree, int message_size){
    float* tmp = (float*)malloc(message_size*sizeof(float));

    for (int i = 0; i<message_size; i++){
        tmp[i] = 12;
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

void allocateLocks(struct Node* tree, int rank){
    int* tmp = (int *) malloc(NUM_BLOCKS*sizeof(int));
    
    for (int i =0; i < NUM_BLOCKS; i++){
        tmp[i] = 0;
    }
    
    cudaSetDevice(rank);

    cudaMalloc((void **)&tree[rank].lock, NUM_BLOCKS*sizeof(int));
    cudaMemcpy(tree[rank].lock, tmp, NUM_BLOCKS*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&tree[rank].ready, NUM_BLOCKS*sizeof(int));
    cudaMemcpy(tree[rank].ready, tmp, NUM_BLOCKS*sizeof(int), cudaMemcpyHostToDevice);
}

void test(struct Node* tree, int rank, int target, int message_size){
    float* tmp = (float*)malloc(message_size*sizeof(float));

    cudaSetDevice(rank); 
    cudaMemcpy(tmp, tree[rank].buffer, message_size*sizeof(float), cudaMemcpyDeviceToHost);

    int n_errs = 0;
    int vals = 0;
    for (int j = 0; j<message_size; j++){
        if (tmp[j]!=target){
            //printf("device %d, index %d. Value %.2f. Error %d\n", rank, j, tmp[j],err);
            n_errs++;
            vals+=tmp[j];
        }
    }
    
    if(n_errs!=0){
        vals = vals/n_errs;
    }
    else{
        vals = target;
    }

    printf("device: %d. Errors: %d. Val: %d\n", rank, n_errs, vals);
    free(tmp);
}
