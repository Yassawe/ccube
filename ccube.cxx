#include "ccube.h"
#include "pthread.h"

//TODO: 
    //1) create N threads, each allocating dummy data and launching allreduce() on its device
    //2) write tests to check functional corectness
    //3) record time performance

int main(){
    struct Node tree[P];
    int num_chunks;


    pthread_t thr[P];

    struct t_args args;

    args.tree = tree;

    createCommunicator(tree);


    for(int i = 0; i<P; i++){
        args.rank = i;
        args.num_chunks=num_chunks;   
        pthread_create(&thr[i], NULL, allreduce, (void *)&args);
    }

    for(int i =0; i<P; i++){
        pthread_join(thr[i], NULL);
    }
}