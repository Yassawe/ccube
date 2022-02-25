#include "ccube.h"
#include "pthread.h"


int main(){
    //create and allocate some data to be allreduced
    // create n threads, each calling allreduce() on its device
    struct Node tree[4];

    createCommunicator(tree);
    
}