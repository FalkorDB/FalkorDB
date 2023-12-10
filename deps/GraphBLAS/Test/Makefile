#-------------------------------------------------------------------------------
# GraphBLAS/Test/Makefile
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

# This Makefile is only intended to clean up the compiled files and files
# created by the make.m function.  To compile the GraphBLAS Tests type 'make'
# in the Command Window.  To run the tests in Test/ type 'testall'.

nothing:
	- echo "no action taken; use 'make clean' or 'make distclean'"

clean:
	- $(RM) -r *.o *.obj *.out

distclean: clean
	$(RM) *.mex* errlog*.txt spok/*.mex* log.txt fprint.txt spok/*/*.mex*

purge: distclean

