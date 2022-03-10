#include "ccube.h"

void* allreduce(void* ptr){

    int* ret = (int*)  malloc(sizeof(int));
    struct t_args* args = (struct t_args*)ptr;
    
    int rank = args->rank;
    int num_chunks = args->num_chunks;
    struct Node* tree = args->tree;

    int parent = tree[rank].parent;
    int child = tree[rank].child;

    *ret = launch(tree, rank, parent, child, num_chunks);

    pthread_exit(ret);
}

int main(int argc, char *argv[]){
    int message_size = 1024; //message size in terms of float32 elements

    struct Node tree[P];
    pthread_t thr[P];
    struct t_args args[P];

    int num_chunks = message_size/CHUNK_SIZE;

    createCommunicator(tree);
    allocateMemoryBuffers(tree, message_size);
    
    //check_p2p();

    testp2p(tree, 0, 2, num_chunks);

    // for(int i = 0; i<P; i++){
    //     args[i].tree = tree;
    //     args[i].rank = i;
    //     args[i].num_chunks=num_chunks;   
    //     pthread_create(&thr[i], NULL, allreduce, (void *)&args[i]);
    // }

    // for(int i =0; i<P; i++){
    //     pthread_join(thr[i], NULL);
    // }

    test(tree, 0, 2, message_size);
    freeMemoryBuffers(tree);
    killCommunicator(tree);
}

