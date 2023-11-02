SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

This is the GraphBLAS/Source/Template folder.

These files are not stand-alone, but are #include'd in files in Source/*.c,
Source/GB.h, Source/Generator/*.c, Source/FactoryKernels/*.c, and
Source/JitKernels/*.c to generate code.

Files with the name GB_*_factory.c are "switch factories", where the
#include'ing file #define's a GB_WORKER macro that is then specialized by the
*factory.c code to generate workers for different semirings, types, and
operators.  Some of the *factory.c files are two-level, where one *factory.c
file #include's another one from this same folder.

