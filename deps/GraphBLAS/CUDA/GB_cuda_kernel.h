//------------------------------------------------------------------------------
// CUDA/GB_cuda_kernel.h: definitions for all GraphBLAS CUDA kernels
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file is #include'd into all CUDA kernels for GraphBLAS.  It provides
// a subset of GraphBLAS.h and GB.h, plus other definitions.

#pragma once

#define GB_CUDA_KERNEL

#undef  ASSERT
#define ASSERT(x)

//------------------------------------------------------------------------------
// TODO: this will be in the jit code:
#define chunksize 128 

//------------------------------------------------------------------------------
// NVIDIA warp size
//------------------------------------------------------------------------------

#define WARPSIZE 32
#define LOG2_WARPSIZE 5

//------------------------------------------------------------------------------

#ifndef INFINITY
#define INFINITY (std::numeric_limits<double>::max())
#endif

// for internal static inline functions
#undef  GB_STATIC_INLINE
#define GB_STATIC_INLINE static __device__ __inline__

//------------------------------------------------------------------------------
// subset of GraphBLAS.h
//------------------------------------------------------------------------------

#ifndef GRAPHBLAS_H
#define GRAPHBLAS_H

#undef restrict
#undef GB_restrict
#define GB_restrict __restrict__
#define restrict GB_restrict

#include <stdint.h>
//#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#undef  GB_GLOBAL
#define GB_GLOBAL extern

// GB_STR: convert the content of x into a string "x"
#define GB_XSTR(x) GB_STR(x)
#define GB_STR(x) #x

#undef  GxB_MAX_NAME_LEN
#define GxB_MAX_NAME_LEN 128

typedef uint64_t GrB_Index ;
typedef struct GB_Descriptor_opaque *GrB_Descriptor ;
typedef struct GB_Type_opaque *GrB_Type ;
typedef struct GB_UnaryOp_opaque *GrB_UnaryOp ;
typedef struct GB_BinaryOp_opaque *GrB_BinaryOp ;
typedef struct GB_IndexUnaryOp_opaque *GrB_IndexUnaryOp ;
typedef struct GB_Monoid_opaque *GrB_Monoid ;
typedef struct GB_Semiring_opaque *GrB_Semiring ;
typedef struct GB_Scalar_opaque *GrB_Scalar ;
typedef struct GB_Vector_opaque *GrB_Vector ;
typedef struct GB_Matrix_opaque *GrB_Matrix ;
typedef struct GB_Context_opaque *GxB_Context ;
typedef struct GB_Global_opaque *GrB_Global ;
typedef struct GB_Iterator_opaque *GxB_Iterator ;

#define GxB_HYPERSPARSE 1   // store matrix in hypersparse form
#define GxB_SPARSE      2   // store matrix as sparse form (compressed vector)
#define GxB_BITMAP      4   // store matrix as a bitmap
#define GxB_FULL        8   // store matrix as full; all entries must be present

typedef void (*GxB_unary_function)  (void *, const void *) ;
typedef void (*GxB_binary_function) (void *, const void *, const void *) ;

typedef bool (*GxB_select_function)      // return true if A(i,j) is kept
(
    GrB_Index i,                // row index of A(i,j)
    GrB_Index j,                // column index of A(i,j)
    const void *x,              // value of A(i,j)
    const void *thunk           // optional input for select function
) ;

typedef void (*GxB_index_unary_function)
(
    void *z,            // output value z, of type ztype
    const void *x,      // input value x of type xtype; value of v(i) or A(i,j)
    GrB_Index i,        // row index of A(i,j)
    GrB_Index j,        // column index of A(i,j), or zero for v(i)
    const void *y       // input scalar y
) ;

#define GxB_GLOBAL_GPU_ID 26

typedef enum
{
    // for all GrB_Descriptor fields:
    GxB_DEFAULT = 0,    // default behavior of the method

    // for GrB_OUTP only:
    GrB_REPLACE = 1,    // clear the output before assigning new values to it

    // for GrB_MASK only:
    GrB_COMP = 2,       // use the structural complement of the input
    GrB_SCMP = 2,       // same as GrB_COMP (historical; use GrB_COMP instead)
    GrB_STRUCTURE = 4,  // use the only pattern of the mask, not its values

    // for GrB_INP0 and GrB_INP1 only:
    GrB_TRAN = 3,       // use the transpose of the input

    // for GxB_AxB_METHOD only:
    GxB_AxB_GUSTAVSON = 1001,   // gather-scatter saxpy method
    GxB_AxB_DOT       = 1003,   // dot product
    GxB_AxB_HASH      = 1004,   // hash-based saxpy method
    GxB_AxB_SAXPY     = 1005    // saxpy method (any kind)
}
GrB_Desc_Value ;

#endif

//------------------------------------------------------------------------------
// subset of GB.h
//------------------------------------------------------------------------------

//#include GB_iceil.h
#define GB_ICEIL(a,b) (((a) + (b) - 1) / (b))
//#include GB_imin.h
#define GB_IMAX(x,y) (((x) > (y)) ? (x) : (y))
#define GB_IMIN(x,y) (((x) < (y)) ? (x) : (y))
//#include GB_zombie.h
#define GB_FLIP(i)             (-(i)-2)
#define GB_IS_FLIPPED(i)       ((i) < 0)
#define GB_IS_ZOMBIE(i)        ((i) < 0)
#define GB_IS_NOT_FLIPPED(i)   ((i) >= 0)
#define GB_UNFLIP(i)           (((i) < 0) ? GB_FLIP(i) : (i))
#define GBI_UNFLIP(Ai,p,avlen)      \
    ((Ai == NULL) ? ((p) % (avlen)) : GB_UNFLIP (Ai [p]))

#include "GB_index.h"
#include "GB_partition.h"
#include "GB_pun.h"
#include "GB_opaque.h"
#include "GB_int64_mult.h"
#define GB_HAS_CMPLX_MACROS 1
#include "GB_complex.h"

// version for the GPU, with fewer branches
#define GB_TRIM_BINARY_SEARCH(i,X,pleft,pright)                             \
{                                                                           \
    /* binary search of X [pleft ... pright] for integer i */               \
    while (pleft < pright)                                                  \
    {                                                                       \
        int64_t pmiddle = (pleft + pright) >> 1 ;                           \
        bool less = (X [pmiddle] < i) ;                                     \
        pleft  = less ? (pmiddle+1) : pleft ;                               \
        pright = less ? pright : pmiddle ;                                  \
    }                                                                       \
    /* binary search is narrowed down to a single item */                   \
    /* or it has found the list is empty */                                 \
    ASSERT (pleft == pright || pleft == pright + 1) ;                       \
}

#define GB_BINARY_SEARCH(i,X,pleft,pright,found)                            \
{                                                                           \
    GB_TRIM_BINARY_SEARCH (i, X, pleft, pright) ;                           \
    found = (pleft == pright && X [pleft] == i) ;                           \
}

#define GB_SPLIT_BINARY_SEARCH(i,X,pleft,pright,found)                      \
{                                                                           \
    GB_BINARY_SEARCH (i, X, pleft, pright, found)                           \
    if (!found && (pleft == pright))                                        \
    {                                                                       \
        if (i > X [pleft])                                                  \
        {                                                                   \
            pleft++ ;                                                       \
        }                                                                   \
        else                                                                \
        {                                                                   \
            pright++ ;                                                      \
        }                                                                   \
    }                                                                       \
}

static __device__ __inline__ int64_t GB_search_for_vector_device
(
    const int64_t p,                // search for vector k that contains p
    const int64_t *restrict Ap,  // vector pointers to search
    int64_t kleft,                  // left-most k to search
    int64_t anvec,                  // Ap is of size anvec+1
    int64_t avlen                   // A->vlen
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (Ap == NULL)
    { 
        // A is full or bitmap
        ASSERT (p >= 0 && p < avlen * anvec) ;
        return ((avlen == 0) ? 0 : (p / avlen)) ;
    }

    // A is sparse
    ASSERT (p >= 0 && p < Ap [anvec]) ;

    //--------------------------------------------------------------------------
    // search for k
    //--------------------------------------------------------------------------

    int64_t k = kleft ;
    int64_t kright = anvec ;
    bool found ;
    GB_SPLIT_BINARY_SEARCH (p, Ap, k, kright, found) ;
    if (found)
    {
        // Ap [k] == p has been found, but if k is an empty vector, then the
        // next vector will also contain the entry p.  In that case, k needs to
        // be incremented until finding the first non-empty vector for which
        // Ap [k] == p.
        ASSERT (Ap [k] == p) ;
        while (k < anvec-1 && Ap [k+1] == p)
        { 
            k++ ;
        }
    }
    else
    { 
        // p has not been found in Ap, so it appears in the middle of Ap [k-1]
        // ... Ap [k], as computed by the binary search.  This is the range of
        // entries for the vector k-1, so k must be decremented.
        k-- ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    // The entry p must reside in a non-empty vector.
    ASSERT (k >= 0 && k < anvec) ;
    ASSERT (Ap [k] <= p && p < Ap [k+1]) ;

    return (k) ;
}

