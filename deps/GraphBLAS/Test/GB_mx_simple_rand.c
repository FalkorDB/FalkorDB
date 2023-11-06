//------------------------------------------------------------------------------
// GraphBLAS/Test/GB_mx_simple_rand.c: a very simple random number generator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

//  The POSIX.1-2001 example of rand, duplicated here so that the same sequence
//  will be generated on different machines.  The purpose is not to generate
//  high-quality random numbers, but to ensure the same sequence is generated
//  on any computer or operating system.  This allows the GraphBLAS tests and
//  demos to be repeatable.

//  Since the simple_rand ( ) function is replicated from the POSIX.1-2001
//  standard, no copyright claim is intended for this specific file.  The
//  copyright statement above applies to all of SuiteSparse:GraphBLAS, not
//  this file.

#include "GB_mex.h"

// simple_rand is not thread-safe
uint64_t simple_rand_next = 1 ;

// return a random number between 0 and SIMPLE_RAND_MAX
uint64_t simple_rand (void)
{
   simple_rand_next = simple_rand_next * 1103515245 + 12345 ;
   return ((simple_rand_next/65536) % (SIMPLE_RAND_MAX + 1)) ;
}

// set the seed
void simple_rand_seed (uint64_t seed)
{
   simple_rand_next = seed ;
}

// get the seed
uint64_t simple_rand_getseed (void)
{
   return (simple_rand_next) ;
}

// return a random uint64_t
uint64_t simple_rand_i ( )
{
    uint64_t i = 0 ;
    for (int k = 0 ; k < 5 ; k++)
    {
        i = SIMPLE_RAND_MAX * i + simple_rand ( ) ;
    }
    return (i) ;
}

// return a random double between 0 and 1, inclusive
double simple_rand_x ( )
{
    return (((double) simple_rand_i ( )) / ((double) UINT64_MAX)) ;
}

