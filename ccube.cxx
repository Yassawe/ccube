#include "ccube.h"
#include "pthread.h"

//TODO: 
    //1) create N threads, each allocating dummy data and launching allreduce() on its device
    //2) write tests to check functional corectness
    //3) record time performance

int main(){
    struct Node tree[P];
    int num_chunks;

    createCommunicator(tree);

    pthread_t thr[P];

    for(int i = 0; i<P; i++){
        struct t_args args = {tree, i, num_chunks}; 
        pthread_create(&thr[i], NULL, allreduce, args);
    }
    //????????????????????????????
    
}