#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "ccube.h"

void allocate_lock(volatile int* pointer, int num_blocks){
    cudaMalloc((void **)&pointer, size*sizeof(int));
    cudaMemset(pointer, 0, size*sizeof(int));
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

    int num_blocks = (CHUNK_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE;

    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(2,0);
    cudaStreamCreateWithFlags(&(tree[0].R_stream), cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&(tree[0].B_stream), cudaStreamNonBlocking);
    tree[0].left = 2;
    tree[0].right = -1;
    tree[0].parent  = -1;
    allocate_lock(tree[0].r_lock, num_blocks);
    allocate_lock(tree[0].b_lock, num_blocks);
    allocate_lock(tree[0].r_done, num_blocks);
    allocate_lock(tree[0].b_done, num_blocks);


    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(2,0);
    cudaStreamCreateWithFlags(&(tree[1].R_stream), cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&(tree[1].B_stream), cudaStreamNonBlocking);
    tree[1].left = -1;
    tree[1].right = -1;
    tree[1].parent = 2;
    allocate_lock(tree[1].r_lock, num_blocks);
    allocate_lock(tree[1].b_lock, num_blocks);
    allocate_lock(tree[1].r_done, num_blocks);
    allocate_lock(tree[1].b_done, num_blocks);


    cudaSetDevice(2);
    cudaDeviceEnablePeerAccess(1,0);
    cudaDeviceEnablePeerAccess(3,0);
    cudaStreamCreateWithFlags(&(tree[2].R_stream), cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&(tree[2].B_stream), cudaStreamNonBlocking);
    tree[2].left = 1;
    tree[2].right = 3;
    tree[2].parent = 0;
    allocate_lock(tree[2].r_lock, num_blocks);
    allocate_lock(tree[2].b_lock, num_blocks);
    allocate_lock(tree[2].r_done, num_blocks);
    allocate_lock(tree[2].b_done, num_blocks);

    
    cudaSetDevice(3);
    cudaDeviceEnablePeerAccess(2,0);
    cudaStreamCreateWithFlags(&(tree[3].R_stream), cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&(tree[3].B_stream), cudaStreamNonBlocking);
    tree[3].left = -1;
    tree[3].right = -1;
    tree[3].parent = 2;
    allocate_lock(tree[3].r_lock, num_blocks);
    allocate_lock(tree[3].b_lock, num_blocks);
    allocate_lock(tree[3].r_done, num_blocks);
    allocate_lock(tree[3].b_done, num_blocks);
}

void killCommunicator(struct Node* tree){
    for(int i=0; i<P; i++){
        cudaSetDevice(i);
        cudaFree(tree[i].r_lock);
        cudaFree(tree[i].b_lock);
        cudaFree(tree[i].r_done);
        cudaFree(tree[i].b_done);
    }
    delete tree;
}


void* allreduce(void* ptr){

    struct t_args* args = (struct t_args*)ptr;
    
    int rank = args->rank;
    struct Node* tree = args->tree;

    int parent = tree[rank].parent;
    int left = tree[rank].left;
    int right = tree[rank].right;

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

    broadcast_kernel<<<(CHUNK_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, tree[rank].B_stream>>>(parent,
                                                                           left,
                                                                           right,
                                                                           tree[rank].buffer,
                                                                           (parent == -1) ? NULL : tree[parent].buffer,
                                                                           tree[rank].b_lock,
                                                                           (left == -1) ? NULL : tree[left].b_lock,
                                                                           (right == -1) ? NULL : tree[right].b_lock,
                                                                           tree[rank].b_done,
                                                                           (parent == -1) ? NULL : tree[parent].b_done,
                                                                           num_chunks);

    cudaDeviceSynchronize();

}

__global__ void reduce_kernel(int parent,
                              int left,
                              int right,
                              float* self_buff, 
                              float* left_buff, 
                              float* right_buff, 
                              volatile int* r_lock_self, 
                              volatile int* r_lock_parent,
                              volatile int* r_done_self,
                              volatile int* r_done_left,
                              volatile int* r_done_right,
                              volatile int* b_lock_left,
                              volatile int* b_lock_right,
                              int num_chunks)
{
    // grid size = number of elements in a chunk
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int gsize = gridDim.x*blockDim.x; 
    int i=0;
    int index = 0;
    //data-independent conditioning, so no branch divergence (?)
    if(parent!=-1){
        // not root
        if(left!=-1 && right!=-1){
            // two children
            for(i = 0; i<num_chunks; i++){
                index = gid + i*gsize;
                while(r_lock_self[blockIdx.x] == 0);
                self_buff[index] = self_buff[index] + left_buff[index] + right_buff[index];
                __syncthreads();
                if(tid == 0){
                    r_lock_self[blockIdx.x] = 0;
                    r_lock_parent[blockIdx.x] = 1;
                }   
                __syncthreads(); //maybe unnecessary
            }
            if(tid==0){
                r_done_left[blockIdx.x] = 1;
                r_done_right[blockIdx.x] = 1;
            }
        }
        else if (left!=-1){
            // one children
            for(i=0; i<num_chunks; i++){
                index = gid + i*gsize;
                while(r_lock_self[blockIdx.x] == 0);
                self_buff[index] = self_buff[index] + left_buff[index];
                __syncthreads();
                if(tid == 0){
                    r_lock_self[blockIdx.x] = 0;
                    r_lock_parent[blockIdx.x] = 1;
                }
                __syncthreads(); //maybe unnecessary
            }
            if (tid ==0){
                r_done_left[blockIdx.x] = 1;
            }
        }
        else{
            //leaf
            if (tid==0){
                while(r_done_self[blockIdx.x] == 0){
                    r_lock_parent[blockIdx.x] = 1;
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
                while(r_lock_self[blockIdx.x] == 0); 
                self_buff[index] = self_buff[index] + left_buff[index] + right_buff[index];
                __syncthreads();
                if(tid == 0){
                    r_lock_self[blockIdx.x] = 0;
                    b_lock_left[blockIdx.x] = 1;
                    b_lock_right[blockIdx.x] = 1;
                } 
                __syncthreads(); //maybe unnecessary
            }
            if(tid==0){
                r_done_left[blockIdx.x] = 1;
                r_done_right[blockIdx.x] = 1;
                r_done_self[blockIdx.x] = 1;
            }
        }
        else if (left!=-1){
            // one child
            for(i=0; i<num_chunks; i++){
                index = gid + i*gsize;
                while(r_lock_self[blockIdx.x] == 0);
                self_buff[index] = self_buff[index] + left_buff[index];
                __syncthreads();
                if(tid == 0){
                    r_lock_self[blockIdx.x] = 0;
                    b_lock_left[blockIdx.x] = 1;
                }
                __syncthreads(); //maybe unnecessary
            }
            if (tid ==0){
                r_done_left[blockIdx.x] = 1;
                r_done_self[blockIdx.x] = 1;
            }
        }
        // if root is a leaf then call the ambulance: 119
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
    // root: do nothing
}

