#include "ccube.h"
#include "pthread.h"


int main(){
    struct Node tree[P];
    pthread_t thr[P];
    
    
    struct t_args args;
    args.tree = tree;

    createCommunicator(tree);
    
    int num_chunks = 5;//random number for now
    //TODO: create some data and allocate it to buffers on devices to be reduced

    for(int i = 0; i<P; i++){
        args.rank = i;
        args.num_chunks=num_chunks;   
        pthread_create(&thr[i], NULL, allreduce, (void *)&args);
    }

    for(int i =0; i<P; i++){
        pthread_join(thr[i], NULL);
    }
    //TODO: create some unit tests to check functional correctness of the code

    killCommunicator(tree);
}