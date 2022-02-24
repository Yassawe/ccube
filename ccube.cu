#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define CHUNK_SIZE 2048 //in float32 elements
#define BLOCK_SIZE 512


void allocate_lock(int* pointer, int num_blocks){
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
    cudaStreamCreateWithFlags(&(tree[0]->R_stream), cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&(tree[0]->B_stream), cudaStreamNonBlocking);
    tree[0]->left = 2;
    tree[0]->right = -1;
    tree[0]->parent  = -1;
    allocate_lock(tree[0]->r_lock, num_blocks);
    allocate_lock(tree[0]->b_lock, num_blocks);
    allocate_lock(tree[0]->r_done, num_blocks);
    allocate_lock(tree[0]->b_done, num_blocks);


    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(2,0);
    cudaStreamCreateWithFlags(&(tree[1]->R_stream), cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&(tree[1]->B_stream), cudaStreamNonBlocking);
    tree[1]->left = -1;
    tree[1]->right = -1;
    tree[1]->parent = 2;
    allocate_lock(tree[1]->r_lock, num_blocks);
    allocate_lock(tree[1]->b_lock, num_blocks);
    allocate_lock(tree[1]->r_done, num_blocks);
    allocate_lock(tree[1]->b_done, num_blocks);


    cudaSetDevice(2);
    cudaDeviceEnablePeerAccess(1,0);
    cudaDeviceEnablePeerAccess(3,0);
    cudaStreamCreateWithFlags(&(tree[2]->R_stream), cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&(tree[2]->B_stream), cudaStreamNonBlocking);
    tree[2]->left = 1;
    tree[2]->right = 3;
    tree[2]->parent = 0;
    allocate_lock(tree[2]->r_lock, num_blocks);
    allocate_lock(tree[2]->b_lock, num_blocks);
    allocate_lock(tree[2]->r_done, num_blocks);
    allocate_lock(tree[2]->b_done, num_blocks);

    
    cudaSetDevice(3);
    cudaDeviceEnablePeerAccess(2,0);
    cudaStreamCreateWithFlags(&(tree[3]->R_stream), cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&(tree[3]->B_stream), cudaStreamNonBlocking);
    tree[3]->left = -1;
    tree[3]->right = -1;
    tree[3]->parent = 2;
    allocate_lock(tree[3]->r_lock, num_blocks);
    allocate_lock(tree[3]->b_lock, num_blocks);
    allocate_lock(tree[3]->r_done, num_blocks);
    allocate_lock(tree[3]->b_done, num_blocks);
}

void killCommunicator(struct Node* tree, int p){
    for(int i=0; i<p; i++){
        cudaSetDevice(i);
        cudaFree(tree[i]->r_lock);
        cudaFree(tree[i]->b_lock);
        cudaFree(tree[i]->r_done);
        cudaFree(tree[i]->b_done);
    }
    delete tree;
}


void allreduce(struct Node* tree, int rank, int num_chunks){
    cudaSetDevice(rank);
    int parent = *tree[rank].parent;
    int left = *tree[rank].left;
    int right = *tree[rank].right;
    
    reduce_kernel<<<(CHUNK_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(parent,
                                                                        left,
                                                                        right,
                                                                        *tree[rank].bu);

}



__global__ void reduce_kernel(int parent,
                              int left,
                              int right,
                              float* self_buff, 
                              float* left_buff, 
                              float* right_buff, 
                              volatile int* r_lock_self, 
                              int* r_lock_parent,
                              volatile int* r_done_self,
                              int* r_done_left,
                              int* r_done_right,
                              int* b_lock_left,
                              int* b_lock_right,
                              int num_chunks)
{
    // grid size = number of elements in a chunk(CHUNK_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE
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
                                 int* b_lock_left,
                                 int* b_lock_right,
                                 int* b_done_self,
                                 int* b_done_parent,
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
            }
            if (tid==0){
                b_done_parent[blockIdx.x] = 1;
                b_done_self[blockIdx.x] = 1;
            }
        }
    }
    // root: do nothing
}

