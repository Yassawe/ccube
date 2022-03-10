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

void createCommunicator(struct Node* tree){
    /*
    simple pipeline for debugging.
    single block for debugging.
    
            0
            |
            1
            |
            2
            |
            3
    */

    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1,0);
    cudaStreamCreateWithFlags(&(tree[0].stream), cudaStreamNonBlocking);
    tree[0].child = 1;
    tree[0].parent  = -1;
    allocateLocks(tree, 0);


    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(2,0);
    cudaDeviceEnablePeerAccess(0,0);
    cudaStreamCreateWithFlags(&(tree[1].stream), cudaStreamNonBlocking);
    tree[1].child = 2;
    tree[1].parent = 0;
    allocateLocks(tree, 1);

    cudaSetDevice(2);
    cudaDeviceEnablePeerAccess(1,0);
    cudaDeviceEnablePeerAccess(3,0);
    cudaStreamCreateWithFlags(&(tree[2].stream), cudaStreamNonBlocking);
    tree[2].child = 3;
    tree[2].parent = 1;
    allocateLocks(tree, 2);

    cudaSetDevice(3);
    cudaDeviceEnablePeerAccess(2,0);
    cudaStreamCreateWithFlags(&(tree[3].stream), cudaStreamNonBlocking);
    tree[3].child = -1;
    tree[3].parent = 2;
    allocateLocks(tree, 3);
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

__global__ void simple_reduce(int parent,
                              int child,
                              volatile int* lock,
                              volatile int* p_lock,
                              volatile int* ready,
                              volatile int* c_ready,
                              float* self_buffer,
                              float* child_buffer,
                              int num_chunks){
    
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    int gid = bid*blockDim.x + tid;
    int gsize = gridDim.x*blockDim.x;
    
    int i = 0; 
    int index = 0;
    
    if (parent ==-1){
        //root
        for(i=0; i<num_chunks; i++){
            index = gsize*i + gid;
            if(tid == 0) c_ready[bid] = 1;
                
            while(lock[bid]==0);
    
            self_buffer[index] = self_buffer[index] + child_buffer[index];
            __syncthreads();
                
            if(tid == 0) lock[bid] = 0;
        }

    }
    else{
        //non-root
        if (child ==-1){
            //leaf
            for(i=0;i<num_chunks; i++){
                while(ready[bid]==0);
                if (tid==0){
                    p_lock[bid] = 1;
                    ready[bid] = 0;
                }
            }
        }
        else{
            //non-leaf
            for(i = 0; i<num_chunks; i++){
                index = gsize*i + gid;
                if(tid == 0) c_ready[bid] = 1;
                
                while(lock[bid]==0);
    
                self_buffer[index] = self_buffer[index] + child_buffer[index];
                __syncthreads();
                
                if(tid == 0) lock[bid] = 0;
                
                while(ready[bid]==0);
    
                if(tid == 0){
                    p_lock[bid] = 1;
                    ready[bid] = 0;
                } 
            }
        }    
        
    }
    
}

int launch(struct Node* tree, int rank, int num_chunks){
    cudaSetDevice(rank);

    int parent = tree[rank].parent;
    int child = tree[rank].child;

    simple_reduce<<<CHUNK_SIZE/BLOCK_SIZE, BLOCK_SIZE, 0, tree[rank].stream>>>(parent,
                                                                               child,
                                                                               tree[rank].lock,
                                                                               (parent==-1) ? NULL : tree[parent].lock,
                                                                               tree[rank].ready,
                                                                               (child==-1) ? NULL : tree[child].ready,
                                                                               tree[rank].buffer,
                                                                               (child==-1) ? NULL : tree[child].buffer,
                                                                               num_chunks);


    CUDAERRORCHECK(cudaDeviceSynchronize());
    return 0;
}
