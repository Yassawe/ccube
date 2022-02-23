#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// TODO: create fixed topology

// implementation should be agnostic to the node and its position

// should generalize the cases of root, 1 children, 2 children, leaf

// overlap reduction and broadcast under multiport simultaneous send-recieve model




// Vanilla C-Cube, no detours, chaining, tricks or any funny business


struct Node {
    cudaStream_t R_stream; 
    cudaStream_t B_stream;

    int parent;
    int left; //order [left, right].
    int right;

    int* lock_reduce;
    int* lock_broadcast;
    
    float *buffer; 
}

struct Node tree[4];


// prototype for 4 node DGX-1|||| MAKES 0 FUCKING SENSE WHEN TOPOLOGY IS FULLY CONNECTED, EXACTLY 0 BENEFIT OVER 2TREE!!!!
void createCommunicator(){
    /*
    Single Tree logical topology
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

    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(2,0);
    cudaStreamCreateWithFlags(&(tree[1].R_stream), cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&(tree[1].B_stream), cudaStreamNonBlocking);
    tree[1].left = -1;
    tree[1].right = -1;
    tree[1].parent = 2;

    cudaSetDevice(2);
    cudaDeviceEnablePeerAccess(1,0);
    cudaDeviceEnablePeerAccess(3,0);
    cudaStreamCreateWithFlags(&(tree[2].R_stream), cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&(tree[2].B_stream), cudaStreamNonBlocking);
    tree[2].left = 1;
    tree[2].right = 3;
    tree[2].parent = 0;
    
    cudaSetDevice(3);
    cudaDeviceEnablePeerAccess(2,0);
    cudaStreamCreateWithFlags(&(tree[3].R_stream), cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&(tree[3].B_stream), cudaStreamNonBlocking);
    tree[3].left = -1;
    tree[3].right = -1;
    tree[3].parent = 2;
}



// define in-place operation for now
void allreduce(void* sendbuff, void* recvbuff, int message_size, int chunk_size){
    // multiprocess function
    // create n threads, each launching reduce_kernel and broadcast_kernel on every device
    // using tree struct
    // number of threads should be equal or close to the chunk size
}


// adapt the C-Cube under the 3 port simultaneous send-recieve model.
// two streams: reduce and broadcast
// tree should be a structure visible and referenceable from the device, not only host controlled

__global__ void reduce_kernel(float* self_buff, 
                              float* left_buff, 
                              float* right_buff, 
                              volatile int* self_lock, 
                              int* parent_lock,
                              int num_chunks)
{

    // grid size = number of elements in a chunk
    
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int gsize = gridDim.x*blockDim.x; 
    int i=0;
    int index = 0;

    //data-independent condition, so no branch divergence
    if(parent_buff){
        // not root
        if(left_buff && right_buff){
            // two children
            for(i = 0; i<num_chunks; i++){
                index = tid + i*gsize;

                while(*self_lock == 0);
                self_buff[index] = self_buff[index] + left_buff[index] + right_buff[index];
                __syncthreads();

                if(tid == 0){
                    *self_lock = 0;
                    *parent_lock = 1;
                } 
                
            }
        }
        else if (left_buff){
            // one children
            for(i=0; i<num_chunks; i++){
                index = tid + i*gsize;
                while(*self_lock == 0);

                self_buff[index] = self_buff[index] + left_buff[index];
                __syncthreads();

                if(tid == 0){
                    *self_lock = 0;
                    *parent_lock = 1;
                }
            }

        }
        else{
            
            *parent_lock = 1;
        }
        // if no children, do nothing
        
    }

    else{
        // root
        // no difference except we don't touch parent (which doesn't exist)

    }
}

__global__ void broadcast_kernel(float* self_buff, float* parent_buff, float* left_buff, float* right_buff, int chunksize){


}

