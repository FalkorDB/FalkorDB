
ROOT=../..
PLATFORM:=$(shell $(ROOT)/bin/platform --os --arch | awk '$$1=$$1' OFS="-")

all:
	gcc -c bb.c $(ROOT)/cetara/diag/gdb.c -fPIC -I $(ROOT) -I$(ROOT)/.. -DDEBUG
	mkdir -p ../$(PLATFORM)
	ar cr ../$(PLATFORM)/libbb.a bb.o gdb.o
