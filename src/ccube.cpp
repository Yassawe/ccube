#include "ccube.h"

void* allreduce(void* ptr){

    int* ret = (int*)  malloc(sizeof(int));
    struct t_args* args = (struct t_args*)ptr;
    
    struct Node* tree = args->tree;
    int rank = args->rank;
    int message_size = args->message_size;

    *ret = launch(tree, rank,  message_size);

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

    createCommunicator(tree);
    allocateMemoryBuffers(tree, message_size);

    for(int i = 0; i<P; i++){
        args[i].tree = tree;
        args[i].rank = i;
        args[i].message_size=message_size;   
        pthread_create(&thr[i], NULL, allreduce, (void *)&args[i]);
    }

    for(int i =0; i<P; i++){
        pthread_join(thr[i], NULL);
    }

    printf("\nAllReduce done\n");

    test(tree, 0, 12*P, message_size);
    test(tree, 2, 12*P, message_size);
    test(tree, 1, 12*P, message_size);
    test(tree, 3, 12*P, message_size);
    
    freeMemoryBuffers(tree);
    killCommunicator(tree);
}

