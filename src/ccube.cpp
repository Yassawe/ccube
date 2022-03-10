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
    int message_size; //message size in terms of float32 elements

    if(argc==2){
        message_size = atoi(argv[1]);
    }
    else if (argc>2){
        printf("too many arguments\n");
        return 1;
    }
    else{
        printf("expected an argument\n");
        return 1;
    }

    struct Node tree[P];
    pthread_t thr[P];
    struct t_args args[P];

    int num_chunks = (message_size+CHUNK_SIZE-1)/CHUNK_SIZE;

    createCommunicator(tree);
    allocateMemoryBuffers(tree, message_size);
    


    for(int i = 0; i<P; i++){
        args[i].tree = tree;
        args[i].rank = i;
        args[i].num_chunks=num_chunks;   
        pthread_create(&thr[i], NULL, allreduce, (void *)&args[i]);
    }

    for(int i =0; i<P; i++){
        pthread_join(thr[i], NULL);
    }

    test(tree, 0, P, message_size);
    test(tree, 1, P, message_size);
    test(tree, 2, P, message_size);
    test(tree, 3, P, message_size);

    freeMemoryBuffers(tree);
    killCommunicator(tree);
}

