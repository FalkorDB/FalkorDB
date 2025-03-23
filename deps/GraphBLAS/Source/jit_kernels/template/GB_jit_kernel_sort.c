//------------------------------------------------------------------------------
// GB_jit_kernel_sort.c: JIT kernel to sort a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "include/GB_cumsum1.h"
#include "include/GB_rand.h"

// sort macros:
#define GB_SORT(func)    GB_jit_kernel_sort_ ## func
#define GB_SORT_UDT      0
#define GB_ADDR(A,i)     ((A) + (i))
#define GB_GETX(x,A,i)   GB_DECLAREC (x) ; GB_GETC (x, A, i, )
#define GB_COPY(A,i,B,j) A [i] = B [j]
#define GB_SIZE          sizeof (GB_C_TYPE)
#define GB_SWAP(A,i,j)   { GB_C_TYPE t = A [i] ; A [i] = A [j] ; A [j] = t ; }
#define GB_LT(less,a,i,b,j)                                 \
{                                                           \
    GB_BINOP (less, a, b, , ) ;     /* less = (a < b) */    \
    if (!less)                                              \
    {                                                       \
        /* check for equality and tie-break on index */     \
        bool more ;                                         \
        GB_BINOP (more, b, a, , ) ;  /* more = (b < a) */   \
        less = (more) ? false : ((i) < (j)) ;               \
    }                                                       \
}

#include "template/GB_sort_template.c"

