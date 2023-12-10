//------------------------------------------------------------------------------
// GraphBLAS/Demo/Include/simple_rand.h: a very simple random number generator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

//  Since the simple_rand ( ) function is replicated from the POSIX.1-2001
//  standard, no copyright claim is intended for this specific file.  The
//  copyright statement above applies to all of SuiteSparse:GraphBLAS, not
//  this file.

#ifndef SIMPLE_RAND_H
#define SIMPLE_RAND_H

#include <stdint.h>

#define SIMPLE_RAND_MAX 32767

// return a random number between 0 and SIMPLE_RAND_MAX
uint64_t simple_rand (void) ;

// set the seed
void simple_rand_seed (uint64_t seed) ;

// get the seed
uint64_t simple_rand_getseed (void) ;

// return a random double between 0 and 1, inclusive
double simple_rand_x ( ) ;

// return a random uint64_t
uint64_t simple_rand_i ( ) ;

#endif
