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

    int num_c;
    int c[2]; //[left, right] order
    int p;
    
    float *buffer; // do i decide on simultaneous send-recieve model? NO!!!
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
    tree[0].num_c = 1;
    tree[0].c[0] = 2;
    tree[0].p  = -1;

    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(2,0);
    cudaStreamCreateWithFlags(&(tree[1].R_stream), cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&(tree[1].B_stream), cudaStreamNonBlocking);
    tree[1].num_c = 0;
    tree[1].p = 2;

    cudaSetDevice(2);
    cudaDeviceEnablePeerAccess(1,0);
    cudaDeviceEnablePeerAccess(3,0);
    cudaStreamCreateWithFlags(&(tree[2].R_stream), cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&(tree[2].B_stream), cudaStreamNonBlocking);
    tree[2].num_c = 2;
    tree[2].c[0] = 1;
    tree[2].c[1] = 3;
    
    cudaSetDevice(3);
    cudaDeviceEnablePeerAccess(2,0);
    cudaStreamCreateWithFlags(&(tree[3].R_stream), cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&(tree[3].B_stream), cudaStreamNonBlocking);
    tree[3].num_c = 0;
    tree[3].p = 2;
}


// adapt the 2 port simultaneous send-recieve model.