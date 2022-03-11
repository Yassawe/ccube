#include "ccube.h"
#include <cuda.h>

void createCommunicator(struct Node* tree){
    /*
    simple tree
    
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
                              volatile int* b_lock_left,
                              volatile int* b_lock_right,
                              volatile int* b_ready,
                              float* self_buff, 
                              float* left_buff, 
                              float* right_buff,
                              int which, // 0 if self is left, 1 if right
                              int num_chunks)
{   
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid*blockDim.x + tid;
    int gsize = gridDim.x*blockDim.x;
    int i=0;
    int index = 0;
    if (parent == -1){
        //root
        if (left == -1 && right == -1){
            //no children
            return;
        }
        else if(right==-1){
            //one child
            for(i=0;i<num_chunks;i++){
                index = gsize*i+gid;
                if(tid==0) r_ready_left[bid] = 1;
                while(r_lock_self[2*bid]==0);
                self_buff[index] = self_buff[index]+left_buff[index];
                __syncthreads();
                if (tid == 0) r_lock_self[2*bid] = 0;
                while(b_ready[2*bid]==0);
                if (tid == 0){
                    b_ready[2*bid] = 0;
                    b_lock_left[bid] = 1;
                }
            }
        }
        else{
            //two children
            for(i=0;i<num_chunks;i++){
                index = gsize*i+gid;
                if (tid == 0){
                    r_ready_left[bid] = 1;
                    r_ready_right[bid] = 1;
                }
                while(r_lock_self[2*bid]==0 || r_lock_self[2*bid+1]==0);
                self_buff[index] = self_buff[index]+left_buff[index]+right_buff[index];
                __syncthreads();
                if (tid == 0){
                    r_lock_self[2*bid] = 0;
                    r_lock_self[2*bid+1] = 0;
                }
                while(b_ready[2*bid]==0 || b_ready[2*bid+1]==0);
                if(tid==0){
                    b_ready[2*bid] = 0;
                    b_ready[2*bid+1] = 0;
                    b_lock_left[bid] = 1;
                    b_lock_right[bid] = 1;
                }
            }
        }
    }
    else{
        //non-root
        if (left == -1 && right == -1){
            //no children
            for(i=0; i<num_chunks; i++){
                while(r_ready[bid]==0);
                if (tid == 0){
                    r_lock_parent[2*bid+which]=1;
                    r_ready[bid] = 0;
                }
            }
        }
        else if(right==-1){
            //one child
            for(i=0; i<num_chunks; i++){
                index = gsize*i+gid;
                if (tid == 0) r_ready_left[bid] = 1;
                while(r_lock_self[2*bid]==0);
                self_buff[index] = self_buff[index]+left_buff[index];
                __syncthreads();
                if (tid == 0) r_lock_self[2*bid] = 0;
                while(r_ready[bid]==0);
                if (tid == 0){
                    r_lock_parent[2*bid+which]=1;
                    r_ready[bid] = 0;
                }
            }
        }
        else{
            //two children
            for(i=0; i<num_chunks; i++){
                index = gsize*i+gid;
                if (tid == 0){
                    r_ready_left[bid] = 1;
                    r_ready_right[bid] = 1;
                } 
                while(r_lock_self[2*bid]==0 || r_lock_self[2*bid+1]==0);
                self_buff[index] = self_buff[index]+left_buff[index]+right_buff[index];
                __syncthreads();
                if (tid == 0){
                    r_lock_self[2*bid] = 0;
                    r_lock_self[2*bid+1] = 0;
                } 
                while(r_ready[bid]==0);
                if (tid == 0){
                    r_lock_parent[2*bid+which]=1;
                    r_ready[bid] = 0;
                }
            }
        }
    }
}

__global__ void broadcast_kernel(int parent,
                                 int left,
                                 int right,
                                 volatile int* b_lock_self,
                                 volatile int* b_lock_left,
                                 volatile int* b_lock_right,
                                 volatile int* b_ready,
                                 volatile int* b_ready_parent,
                                 float* self_buff,
                                 float* parent_buff,
                                 int which,
                                 int num_chunks){

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid*blockDim.x + tid;
    int gsize = gridDim.x*blockDim.x;
    int i=0;
    int index = 0;
    if (parent==-1){
        //root
        return; //root does nothing
    }
    else{
        // non-root
        if(left == -1 && right == -1){
            // no children
            for(i=0; i<num_chunks; i++){
                index = gsize*i + gid;
                if (tid == 0) b_ready_parent[2*bid+which] = 1;
                while(b_lock_self[bid]==0);
                self_buff[index] = parent_buff[index];
                __syncthreads();
                if (tid == 0) b_lock_self[bid] = 0;
            }
        }
        else if (right == -1){
            // one child
            for(i=0;i<num_chunks;i++){
                index = gsize*i+gid;
                if (tid == 0) b_ready_parent[2*bid+which] = 1;
                while(b_lock_self[bid]==0);
                self_buff[index] = parent_buff[index];
                __syncthreads();
                if (tid == 0) b_lock_self[bid] = 0;
                while(b_ready[2*bid]==0);
                if(tid == 0){
                    b_ready[2*bid] = 0;
                    b_lock_left[bid] = 1;
                }
            }
        }
        else{
            // two children
            for(i=0;i<num_chunks;i++){
                index = gsize*i+gid;
                if (tid == 0) b_ready_parent[2*bid+which] = 1;
                while(b_lock_self[bid]==0);
                self_buff[index] = parent_buff[index];
                __syncthreads();
                if (tid == 0) b_lock_self[bid] = 0;
                while(b_ready[2*bid]==0 || b_ready[2*bid+1]==0);
                if(tid == 0){
                    b_ready[2*bid] = 0;
                    b_ready[2*bid+1] = 0;
                    b_lock_left[bid] = 1;
                    b_lock_right[bid] = 1;
                }
            }
        }
    }
}


int launch(struct Node* tree, int rank, int num_chunks){
    cudaSetDevice(rank);
    int parent = tree[rank].parent;
    int left = tree[rank].left;
    int right = tree[rank].right;
    int which = 0;

    if (parent != -1){
        which = (rank == tree[parent].right) ? 1 : 0;
    }
 
    reduce_kernel<<<NUM_BLOCKS, BLOCK_SIZE, 0, tree[rank].R_stream>>>(parent,
                                                                    left,
                                                                    right, 
                                                                    tree[rank].r_lock,
                                                                    (parent!=-1) ? tree[parent].r_lock : NULL,
                                                                    tree[rank].r_ready,
                                                                    (left!=-1) ? tree[left].r_ready : NULL,
                                                                    (right!=-1) ? tree[right].r_ready : NULL,
                                                                    (left!=-1) ? tree[left].b_lock : NULL,
                                                                    (right!=-1) ? tree[right].b_lock : NULL,
                                                                    tree[rank].b_ready,
                                                                    tree[rank].buffer,
                                                                    (left!=-1) ? tree[left].buffer : NULL,
                                                                    (right!=-1) ? tree[right].buffer : NULL,
                                                                    which,
                                                                    num_chunks);

    broadcast_kernel<<<NUM_BLOCKS, BLOCK_SIZE, 0, tree[rank].B_stream>>>(parent,
                                                                        left,
                                                                        right,
                                                                        tree[rank].b_lock,
                                                                        (left!=-1) ? tree[left].b_lock : NULL,
                                                                        (right!=-1) ? tree[right].b_lock : NULL,
                                                                        tree[rank].b_ready,
                                                                        (parent!=-1) ? tree[parent].b_ready : NULL,
                                                                        tree[rank].buffer,
                                                                        (parent!=-1) ? tree[parent].buffer : NULL,
                                                                        which,
                                                                        num_chunks);
    
    cudaDeviceSynchronize();
    return 0;
}
