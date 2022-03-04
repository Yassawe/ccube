#include "ccube.h"

void* allreduce(void* ptr){

    int* ret = (int*)  malloc(sizeof(int));
    struct t_args* args = (struct t_args*)ptr;
    
    int rank = args->rank;
    int num_chunks = args->num_chunks;
    struct Node* tree = args->tree;

    int parent = tree[rank].parent;
    int left = tree[rank].left;
    int right = tree[rank].right;

    *ret = launch(tree, rank, parent, left, right, num_chunks);

    pthread_exit(ret);
}

int main(int argc, char *argv[]){
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
    free(thr);
    free(&args);
}

