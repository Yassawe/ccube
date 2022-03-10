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
    prototype for 4 node DGX-1
    
            0
            |
            2
           / \
          1   3
    */

    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(2,0);
    cudaStreamCreateWithFlags(&(tree[0].R_stream), cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&(tree[0].B_stream), cudaStreamNonBlocking);
    tree[0].left = 2;
    tree[0].right = -1;
    tree[0].parent  = -1;
    allocateLocks(tree, 0);


    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(2,0);
    cudaStreamCreateWithFlags(&(tree[1].R_stream), cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&(tree[1].B_stream), cudaStreamNonBlocking);
    tree[1].left = -1;
    tree[1].right = -1;
    tree[1].parent = 2;
    allocateLocks(tree, 1);


    cudaSetDevice(2);
    cudaDeviceEnablePeerAccess(0,0);
    cudaDeviceEnablePeerAccess(1,0);
    cudaDeviceEnablePeerAccess(3,0);
    cudaStreamCreateWithFlags(&(tree[2].R_stream), cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&(tree[2].B_stream), cudaStreamNonBlocking);
    tree[2].left = 1;
    tree[2].right = 3;
    tree[2].parent = 0;
    allocateLocks(tree, 2);

    
    cudaSetDevice(3);
    cudaDeviceEnablePeerAccess(2,0);
    cudaStreamCreateWithFlags(&(tree[3].R_stream), cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&(tree[3].B_stream), cudaStreamNonBlocking);
    tree[3].left = -1;
    tree[3].right = -1;
    tree[3].parent = 2;
    allocateLocks(tree, 3);
}

void killCommunicator(struct Node* tree){
    for(int i=0; i<P; i++){
        cudaSetDevice(i);
        cudaFree(tree[i].r_lock);
        cudaFree(tree[i].b_lock);
        cudaFree(tree[i].r_ready);
        cudaFree(tree[i].b_ready);
        cudaStreamDestroy(tree[i].R_stream);
        cudaStreamDestroy(tree[i].B_stream);

        for (int j = 0; j<P; j++){
            cudaDeviceDisablePeerAccess(j);
        }
    }
}

__global__ void reduce_kernel(int parent,
                              int left,
                              int right, 
                              volatile int* r_lock_self, 
                              volatile int* r_lock_parent,
                              volatile int* r_ready,
                              volatile int* r_ready_left,
                              volatile int* r_ready_right,
                              float* self_buff, 
                              float* left_buff, 
                              float* right_buff,
                              int num_chunks)
{
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid*blockDim.x + tid;
    int gsize = gridDim.x*blockDim.x;

    int i=0;
    int index = 0;

    if(parent!=-1){
        // not root

        if(left!=-1 && right!=-1){
            // two children
            for(i = 0; i<num_chunks; i++){
                
                index = gid + i*gsize;

                if (tid==0){
                    r_ready_left[bid]=1;
                    r_ready_right[bid]=1;
                }

                while(r_lock_self[bid] != 2);

                self_buff[index] = self_buff[index] + left_buff[index] + right_buff[index];
                __syncthreads();
                
                if (tid == 0) r_lock_self[bid]=0;
                
                while(r_ready[bid]==0);

                if(tid == 0){
                    atomicAdd(&r_lock_parent[bid], 1);
                    r_ready[bid]=0;
                }    
            }
        }
        else if (left!=-1){
            // one children
            for(i=0; i<num_chunks; i++){
                index = gid + i*gsize;

                if (tid == 0) r_ready_left[bid]=1;

                while(r_lock_self[bid] != 1);

                self_buff[index] = self_buff[index] + left_buff[index];
                __syncthreads();

                if (tid == 0) r_lock_self[bid]==0;

                while(r_ready[bid]==0);

                if(tid == 0){
                    atomicAdd(&r_lock_parent[bid], 1);
                    r_ready[bid]=0;
                }
            }
        }
        else{
            //leaf
            for (i=0; i<num_chunks; i++){
                
                while(r_ready[bid]==0);

                if(tid==0){
                    atomicAdd(&r_lock_parent[bid], 1);
                    r_ready[bid] = 0;
                }
            }
        }
    }
    else{
        // root
        if(left!=-1 && right!=-1){
            // two children
            for(i = 0; i<num_chunks; i++){
                
                index = gid + i*gsize;

                if (tid==0){
                    r_ready_left[bid]=1;
                    r_ready_right[bid]=1;
                }

                while(r_lock_self[bid] == 0);

                self_buff[index] = self_buff[index] + left_buff[index] + right_buff[index];
                __syncthreads();
                
                if (tid == 0) r_lock_self[bid]=0;   
            }
        }
        else if (left!=-1){
            // one child
            for(i=0; i<num_chunks; i++){
                index = gid + i*gsize;

                if (tid == 0) r_ready_left[bid]=1;

                while(r_lock_self[bid] == 0);

                self_buff[index] = self_buff[index] + left_buff[index];
                __syncthreads();

                if (tid == 0) r_lock_self[bid]==0;
            }
        }
        // if root is a leaf then call the ambulance: 119
    }  
}



__global__ void broadcast_kernel(int parent,
                                 int left,
                                 int right,
                                 float* self_buff,
                                 float* parent_buff,
                                 volatile int* b_lock_self,
                                 volatile int* b_lock_left,
                                 volatile int* b_lock_right,
                                 volatile int* b_done_self,
                                 volatile int* b_done_parent,
                                 int num_chunks)
{
    // grid size = num of elements in a chunk
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int gsize = gridDim.x*blockDim.x; 
    int i=0;
    int index = 0;
    if(parent!=-1){
        //not root
        if(left!=-1 && right!=-1){
            // two children
            for (i=0; i<num_chunks; i++){
                index = gid + i*gsize;
                while(b_lock_self[blockIdx.x] == 0);
                self_buff[index] = parent_buff[index];
                __syncthreads();
                if (tid==0){
                    b_lock_self[blockIdx.x] = 0;
                    b_lock_left[blockIdx.x] = 1;
                    b_lock_right[blockIdx.x] = 1;
                }
                __syncthreads(); //maybe unnecessary
            }
            if (tid==0){
                b_done_parent[blockIdx.x] = 1;
            }
        }
        else if (left!=-1){
            // one child
            for (i=0; i<num_chunks; i++){
                index = gid + i*gsize;
                while(b_lock_self[blockIdx.x] == 0);
                self_buff[index] = parent_buff[index];
                __syncthreads();
                if (tid==0){
                    b_lock_self[blockIdx.x] = 0;
                    b_lock_left[blockIdx.x] = 1;
                }
                __syncthreads(); //maybe unnecessary
            }
            if (tid==0){
                b_done_parent[blockIdx.x] = 1;
            }
        }
        else{
            // leaf
            for (i=0; i<num_chunks; i++){
                index = gid + i*gsize;
                while(b_lock_self[blockIdx.x] == 0);
                self_buff[index] = parent_buff[index];
                __syncthreads();
                if (tid==0){
                    b_lock_self[blockIdx.x] = 0;
                }
                __syncthreads(); //maybe unnecessary
            }
            if (tid==0){
                b_done_parent[blockIdx.x] = 1;
                b_done_self[blockIdx.x] = 1;
            }
        }
    }
}

int launch(struct Node* tree, int rank, int num_chunks){

    cudaSetDevice(rank);

    int parent = tree[rank].parent;
    int left = tree[rank].left;
    int right = tree[rank].right;
 
    reduce_kernel<<<(CHUNK_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, tree[rank].R_stream>>>(parent,
                                                                                                left,
                                                                                                right, 
                                                                                                tree[rank].r_lock,
                                                                                                (parent!=-1) ? tree[parent].r_lock : NULL,
                                                                                                tree[rank].r_ready,
                                                                                                (left!=-1) ? tree[left].r_ready : NULL,
                                                                                                (right!=-1) ? tree[right].r_ready : NULL,
                                                                                                tree[rank].buffer,
                                                                                                (left!=-1) ? tree[left].buffer : NULL,
                                                                                                (right!=-1) ? tree[right].buffer : NULL,
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
