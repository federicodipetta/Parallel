# Makefile per il progetto Fuzzy C-Means CPU/GPU
#
# Schema di compilazione:
#   - main.cpp, dataset.cpp, fcm_cpu.cpp  -> compilati con g++ (puro C++,
#     nessuna dipendenza CUDA: fcm_gpu.cuh espone solo una firma di funzione)
#   - fcm_gpu.cu                          -> compilato con nvcc
#   - link finale                         -> con nvcc (necessario per
#     risolvere i simboli della libreria runtime CUDA, cudart)
#
# Se la tua GPU ha una compute capability diversa, aggiorna ARCH di conseguenza.
# Esempi: sm_61 (Pascal, GTX 10xx), sm_75 (Turing, RTX 20xx),
#         sm_86 (Ampere, RTX 30xx), sm_89 (Ada, RTX 40xx)
ARCH ?= sm_75

CXX      = g++
NVCC     = nvcc
CXXFLAGS = -O3 -std=c++17 -Wall
NVCCFLAGS= -O3 -std=c++17 -arch=$(ARCH)

TARGET = fcm_fuzzy

OBJS = main.o dataset.o fcm_cpu.o #fcm_gpu.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

main.o: main.cpp dataset.h fcm_cpu.h #fcm_gpu.cuh
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

dataset.o: dataset.cpp dataset.h
	$(CXX) $(CXXFLAGS) -c dataset.cpp -o dataset.o

fcm_cpu.o: fcm_cpu.cpp fcm_cpu.h
	$(CXX) $(CXXFLAGS) -c fcm_cpu.cpp -o fcm_cpu.o

# fcm_gpu.o: fcm_gpu.cu fcm_gpu.cuh
# 	$(NVCC) $(NVCCFLAGS) -c fcm_gpu.cu -o fcm_gpu.o

clean:
	rm -f *.o $(TARGET)

.PHONY: all clean
