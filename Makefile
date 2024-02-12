#-------------------------------------------------------------------------------
# FalkorDB/Makefile
#-------------------------------------------------------------------------------

# simple Makefile for FalkorDB, relies on cmake to do the actual build.

JOBS ?= 1

default: library

library:
	( cd build && cmake $(CMAKE_OPTIONS) .. && cmake --build . --config Release -j${JOBS} )

# compile with -g
debug:
	( cd build && cmake -DCMAKE_BUILD_TYPE=Debug $(CMAKE_OPTIONS) .. && cmake --build . --config Debug -j$(JOBS) )

# remove all files
clean:
	- rm -rf build/*
