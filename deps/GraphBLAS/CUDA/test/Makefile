#-------------------------------------------------------------------------------
# GraphBLAS/CUDA/Makefile
#-------------------------------------------------------------------------------

# cuda 10.1+ is assumed

all: cudaTest


LIBS = -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lpthreads -lcudadevrt -lcudart -lnvrtc
INC += -I$(CUDA_DIR)/include -I../ -I../../Source -I../../Include -I../../Source/Template -I$(TEMPLATE_DIR) -Igoogletest/include

CUDA_OPTS = -O2 --cudart=shared --gpu-architecture=compute_75\
        --relocatable-device-code true --device-c\
        --std=c++17 -Xcompiler -fPIC

%.o: %.cu
	nvcc -c $(I) $(CUDA_OPTS) $(INC)  -o $@ $< 

config:
	nvidia-smi
	nvcc --version
	@echo " "
	@echo "SO_NAME:   " $(SO_NAME)
	@echo "SO_OPTS:   " $(SO_OPTS)
	@echo "LIBS:      " $(LIBS)
	@echo "CUDA_OPTS: " $(CUDA_OPTS)
	@echo "SRC:       " $(SRC)
	@echo "OBJ:       " $(OBJ)
	@echo "I:         " $(I)
	@echo " "
	gcc  --version
	icc  --version

clean:
	rm -f *.o
	rm -f stringify
	rm -f cudaTest 
	rm -f testJit
.PHONY: clean

distclean: clean
	rm -f *.so *.a

purge: distclean

################################################################################

GXX     ?= g++
GCC     ?= gcc
DOXYGEN ?= doxygen
CXXFLAGS ?= -O3 -Wall -g -fmessage-length=80
CFLAGS ?= -O2  -g -std=c11 

CXX11 ?= 1

CUDA_DIR ?= /usr/local/cuda

CXXFLAGS += -pthread

ifeq ($(CXX11),1)
	CXXFLAGS += -std=c++17
endif

EMBED_BEGIN = -rdynamic -Wl,-b,binary,
EMBED_END   = ,-b,default

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CXXFLAGS += -D LINUX 
	CUDA_LIB_DIR = $(CUDA_DIR)/lib64
else ifeq ($(UNAME_S),Darwin)
	CUDA_LIB_DIR = $(CUDA_DIR)/lib
endif

TEMPLATE_DIR ?= ../templates

LIB += -ldl -L$(CUDA_LIB_DIR) -L$(CUDA_LIB_DIR)/stubs -lcuda -lcudadevrt -lcudart -lnvrtc

HEADERS = jitify.hpp dataFactory.hpp jitFactory.hpp jitTestFactory.hpp semiringFactory.hpp \
          ../type_name.hpp 

TEMPLATES :=  $(wildcard $(TEMPLATE_DIR)/*.cu)

CU_OBJS := ../GB_jit_cache.o ../GB_jit_launcher.o 

CFILES := $(wildcard ../*.c)  

COBJS := $(patsubst %.c, %.o, $(CFILES) )

JIT_TEMP := $(patsubst %.cu, %.cu.jit, $(TEMPLATES))

GTEST_LIB := googletest/build/lib/libgtest.a googletest/build/lib/libgtest_main.a

%.cu.jit: %.cu 
	../stringify $? > $@

stringify: stringify.cpp
	$(GXX) -o $@ $< -O3 -Wall

%.o: %.c
	$(GXX) -c -o $@ $< $(CFLAGS) $(INC) 

%.o: %.cpp
	$(GXX) -c -o $@ $< $(CXXFLAGS) $(INC) 

cu_link.o: $(CU_OBJS)
	nvcc --gpu-architecture=compute_75 --device-link $(CU_OBJS)  --output-file cu_link.o


testJit: ../tofix/testJit.cpp $(OBJS) $(HEADERS) $(JIT_TEMP)
	$(GXX) -o $@ $< $(CXXFLAGS) $(INC) $(OBJS) $(LIB)

AxB_dot3_test_instances.hpp:  testGen.py
	python3 testGen.py


instances :=  AxB_dot3_test_instances.hpp 


cudaTest: cudaTest.cpp.bak $(COBJS) $(OBJS) $(HEADERS) $(JIT_TEMP) cu_link.o AxB_dot3_cuda_tests.hpp.bak  $(instances)
	$(GXX) -o $@ $< $(CXXFLAGS) $(INC) $(COBJS) $(CU_OBJS) cu_link.o $(LIB) $(GTEST_LIB)

%.cu: %.cutmp
	cp $? $@


doc: jitify.hpp Doxyfile
	$(DOXYGEN) Doxyfile
.PHONY: doc


