#-------------------------------------------------------------------------------
# GraphBLAS/Tcov/Makefile
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

UNAME := $(shell uname)

# pick your compiler:
# CC = gcc
  CC = cc
# CC = clang
# CC = xlc
# CC = gcc-8
# CC = gcc-11
# CC = icc
ifeq ($(UNAME),Darwin)
    CFLAGS  = -g -Xclang -fopenmp -std=c11 -fPIC -Wno-pragmas -fno-common
    LDFLAGS = -Xclang -fopenmp
else
    CFLAGS  = -g -std=c11 -fopenmp -fexceptions -fPIC -Wno-pragmas
    LDFLAGS = -fopenmp
endif

SRC = tmp_source/*
INC = tmp_include/*
SRC2 = $(notdir $(wildcard $(SRC)))
OBJ = $(SRC2:.c=.o)
LIBS = 
I = -Itmp_include -I../Test -I../rmm_wrap -I../zstd -I../zstd/zstd_subset -I../lz4 -I../xxHash -I../Source/JitKernels
CPPFLAGS = $(I) -DGBNCPUFEAT -DGBCOVER $(MATLAB)
SO_OPTS = $(LDFLAGS)

ifeq ($(UNAME),Darwin)
    # Mac
    SO_NAME = libgraphblas_tcov.dylib
    SO_OPTS += -dynamiclib -shared -Wl,-install_name -Wl,$(SO_NAME) -undefined dynamic_lookup
else
    # Linux
    SO_NAME = libgraphblas_tcov.so
    SO_OPTS += -shared -Wl,-soname -Wl,$(SO_NAME) -Wl,--no-undefined
endif

default: $(SO_NAME)

%.o: tmp_source/%.c $(INC)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $(notdir $@)

$(SO_NAME): $(OBJ)
	$(CC) $(SO_OPTS) $^ $(LIBS) -o $@ -lm -ldl

.KEEP: $(OBJ)

clean:
	$(RM) *.o *.d

distclean: clean
	$(RM) *.mex* tmp*/* errlog*.txt grbstat.mat fprint.txt
	$(RM) libgraphblas_tcov.*

purge: distclean
	$(RM) log.txt

config:
	@echo "CFLAGS:   " $(CFLAGS)
	@echo "CPPFLAGS: " $(CPPFLAGS)
	@echo "LDFLAGS:  " $(LDFLAGS)
