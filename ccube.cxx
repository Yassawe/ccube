#include "ccube.h"
#include "pthread.h"

#define P 4

int main(){
    struct Node tree[P];

    createCommunicator(tree);

    pthread_t thr[P];

    //TODO: 
    //1) create N threads, each allocating dummy data and launching allreduce() on its device
    //2) write tests to check functional corectness
    //3) record time performance
}