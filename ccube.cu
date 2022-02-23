#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// TODO: create fixed topology

// implementation should be agnostic to the node and its position

// should generalize the cases of root, 1 children, 2 children, leaf

// overlap reduction and broadcast under multiport simultaneous send-recieve model

#define CHUNK_SIZE 2048 //in float32 elements
#define BLOCK_SIZE 512


// Vanilla C-Cube, no detours, chaining, tricks or any funny business


struct Node {
    cudaStream_t R_stream; 
    cudaStream_t B_stream;

    int parent;
    int left; //order [left, right].
    int right;

    int* r_lock;
    int* b_lock;

    int* r_done;
    int* b_done;

    
    float *buffer; 
}

struct Node tree[4];


void allocate_lock(int* pointer, int num_blocks){
    cudaMalloc((void **)&pointer, size*sizeof(int));
    cudaMemset(pointer, 0, size*sizeof(int));
}

// prototype for 4 node DGX-1
void createCommunicator(){
    /*
    Single Tree logical topology
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



// define in-place operation for now
void allreduce(void* buff, int message_size, int chunk_size){
    // multiprocess function
    // create n threads, each launching reduce_kernel and broadcast_kernel on every device
    // using tree struct
    // number of threads should be equal or close to the chunk size
}


// adapt the C-Cube under the 3 port simultaneous send-recieve model.
// two streams: reduce and broadcast
// tree should be a structure visible and referenceable from the device, not only host controlled

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
                while(r_lock_self[blockIdx.x] == 0); //TODO: make each lock block dependent, for global sync
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

