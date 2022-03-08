CUDA_PATH=/usr/local/cuda

CC=g++ -m64
CC_FLAGS=-Wall -pthread
CC_LIBS= 

NVCC=nvcc
NVCC_FLAGS=-m64 --gpu-architecture compute_70
NVCC_LIBS=

CUDA_LIB_DIR= -L$(CUDA_PATH)/lib64
CUDA_INC_DIR= -I$(CUDA_PATH)/include
CUDA_LINK_LIBS= -lcudart

EXE = ccube

# Link C and CUDA compiled object files to target executable:

$(EXE) : bin/main.o bin/memory.o bin/cuda.o
	$(CC) $(CC_FLAGS) bin/main.o bin/memory.o bin/cuda.o -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)


# Compile C++ source files to object files:
bin/main.o : src/ccube.cpp src/ccube.h
	$(CC) $(CC_FLAGS) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) -c $< -o $@

bin/memory.o : src/memory.cpp src/ccube.h
	$(CC) $(CC_FLAGS) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) -c $< -o $@

# Compile CUDA source files to object files:
bin/cuda.o : src/ccube.cu src/ccube.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

clean:
	$(RM) $(EXE) bin/* *.o 
