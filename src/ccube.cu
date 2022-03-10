#include "ccube.h"
#include <cuda.h>

#define CUDAERRORCHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void allocate_lock(int* pointer){
    cudaMalloc((void **)&pointer, sizeof(int));
    cudaMemset(pointer, 0, sizeof(int));
}

void createCommunicator(struct Node* tree){
    /*
    simple pipeline for debugging.
    single block for debugging.
    
            0
            |
            1
            |
            2
    */

    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1,0);
    cudaStreamCreateWithFlags(&(tree[0].stream), cudaStreamNonBlocking);
    tree[0].child = 1;
    tree[0].parent  = -1;
    allocate_lock(tree[0].lock);
    allocate_lock(tree[0].ready);


    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(2,0);
    cudaDeviceEnablePeerAccess(0,0);
    cudaStreamCreateWithFlags(&(tree[1].stream), cudaStreamNonBlocking);
    tree[1].child = 2;
    tree[1].parent = 0;
    allocate_lock(tree[1].lock);
    allocate_lock(tree[1].ready);

    cudaSetDevice(2);
    cudaDeviceEnablePeerAccess(1,0);
    cudaStreamCreateWithFlags(&(tree[2].stream), cudaStreamNonBlocking);
    tree[2].child = -1;
    tree[2].parent = 1;
    allocate_lock(tree[2].lock);
    allocate_lock(tree[2].ready);

}

void killCommunicator(struct Node* tree){
    for(int i=0; i<P; i++){
        cudaSetDevice(i);
        cudaFree(tree[i].lock);
        cudaFree(tree[i].ready);
        cudaStreamDestroy(tree[i].stream);
        for (int j = 0; j<P; j++){
            cudaDeviceDisablePeerAccess(j);
        }
    }
}



int launch(struct Node* tree, int rank, int parent, int left, int right, int num_chunks){

    cudaSetDevice(rank);

 
    reduce_kernel<<<(CHUNK_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, tree[rank].R_stream>>>(parent,
                                                                                                left,
                                                                                                right,
                                                                                                tree[rank].buffer,
                                                                                                (left == -1) ? NULL : tree[left].buffer,
                                                                                                (right == -1) ? NULL : tree[right].buffer,
                                                                                                tree[rank].r_lock,
                                                                                                (parent == -1) ? NULL : tree[parent].r_lock,
                                                                                                tree[rank].r_done,
                                                                                                (left == -1) ? NULL : tree[left].r_done,
                                                                                                (right == -1) ? NULL : tree[right].r_done,
                                                                                                (left == -1) ? NULL : tree[left].b_lock,
                                                                                                (right == -1) ? NULL : tree[right].b_lock,
                                                                                                num_chunks);

    CUDAERRORCHECK(cudaDeviceSynchronize());

    // broadcast_kernel<<<(CHUNK_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, tree[rank].B_stream>>>(parent,
    //                                                                                                 left,
    //                                                                                                 right,
    //                                                                                                 tree[rank].buffer,
    //                                                                                                 (parent == -1) ? NULL : tree[parent].buffer,
    //                                                                                                 tree[rank].b_lock,
    //                                                                                                 (left == -1) ? NULL : tree[left].b_lock,
    //                                                                                                 (right == -1) ? NULL : tree[right].b_lock,
    //                                                                                                 tree[rank].b_done,
    //                                                                                                 (parent == -1) ? NULL : tree[parent].b_done,
    //                                                                                                 num_chunks);
    return 0;
}


//[DEBUG]

__global__ void p2p_sum(float* a, float* b, int num_chunks){
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int gsize = gridDim.x*blockDim.x;

    int i = 0;
    int index = 0;

    for (i=0; i<num_chunks; i++){
        index = gid + i*gsize;
        a[index] = a[index] + b[index];
        __syncthreads();
    }
}

void testp2p(struct Node* tree,int rank, int peer, int num_chunks){
    cudaSetDevice(rank);
    p2p_sum<<<CHUNK_SIZE/BLOCK_SIZE, BLOCK_SIZE>>>(tree[rank].buffer, tree[peer].buffer, num_chunks);
    CUDAERRORCHECK(cudaDeviceSynchronize());
}