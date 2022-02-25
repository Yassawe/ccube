#include "ccube.h"
#include "pthread.h"


int main(){
    struct Node tree[4];

    createCommunicator(tree);
    //TODO: 
    //1) create N threads, each allocating dummy data and launching allreduce() on its device
    //2) write tests to check functional corectness
    //3) record time performance
}