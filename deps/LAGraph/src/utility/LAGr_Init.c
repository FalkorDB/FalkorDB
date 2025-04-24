//------------------------------------------------------------------------------
// LAGr_Init: start GraphBLAS and LAGraph, and set malloc/etc functions
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2023 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

#define LG_FREE_ALL ;

#include "LG_internal.h"

//------------------------------------------------------------------------------
// LG_LAGr_Init_has_been_called: a static value only accessible within this file
//------------------------------------------------------------------------------

// LG_LAGr_Init_has_been_called indicates if LAGr_Init has been called.
// LAGr_Init (or LAGraph_Init) can be called only once, even after
// LAGraph_Finalize has been called.  For testing purposes, the flag can be
// cleared by src/test/test_Xinit, to allow LAGr_Init or LAGraph_Init to be
// called again.

static bool LG_LAGr_Init_has_been_called = false ;

// LG_LAGr_Init_has_been_called is only modified or accessed by the following
// two routines (even in this file).  The two functions are made accessible via
// LAGRAPH_PUBLIC, only for testing purposes by src/test/test_Xinit.c.

LAGRAPH_PUBLIC void LG_set_LAGr_Init_has_been_called (bool setting) ;
LAGRAPH_PUBLIC bool LG_get_LAGr_Init_has_been_called (void) ;

LAGRAPH_PUBLIC
void LG_set_LAGr_Init_has_been_called (bool setting)
{
    LG_LAGr_Init_has_been_called = setting ;
}

LAGRAPH_PUBLIC
bool LG_get_LAGr_Init_has_been_called (void)
{
    return (LG_LAGr_Init_has_been_called) ;
}

//------------------------------------------------------------------------------
// LAGraph global objects
//------------------------------------------------------------------------------

// LAGraph_plus_first_T: using the GrB_PLUS_MONOID_T monoid and the
// corresponding GrB_FIRST_T multiplicative operator.
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_first_int8   = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_first_int16  = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_first_int32  = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_first_int64  = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_first_uint8  = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_first_uint16 = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_first_uint32 = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_first_uint64 = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_first_fp32   = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_first_fp64   = NULL ;

// LAGraph_plus_second_T: using the GrB_PLUS_MONOID_T monoid and the
// corresponding GrB_SECOND_T multiplicative operator.
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_second_int8   = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_second_int16  = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_second_int32  = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_second_int64  = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_second_uint8  = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_second_uint16 = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_second_uint32 = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_second_uint64 = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_second_fp32   = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_second_fp64   = NULL ;

// LAGraph_plus_one_T: using the GrB_PLUS_MONOID_T monoid and the
// corresponding GrB_ONEB_T multiplicative operator.
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_one_int8   = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_one_int16  = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_one_int32  = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_one_int64  = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_one_uint8  = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_one_uint16 = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_one_uint32 = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_one_uint64 = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_one_fp32   = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_plus_one_fp64   = NULL ;


// use LAGraph_any_one_bool, etc

// LAGraph_any_one_T: using the GrB_MIN_MONOID_T for non-boolean types
// or GrB_LOR_MONOID_BOOL for boolean, and the GrB_ONEB_T multiplicative op.
LAGRAPH_PUBLIC GrB_Semiring LAGraph_any_one_bool   = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_any_one_int8   = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_any_one_int16  = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_any_one_int32  = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_any_one_int64  = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_any_one_uint8  = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_any_one_uint16 = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_any_one_uint32 = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_any_one_uint64 = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_any_one_fp32   = NULL ;
LAGRAPH_PUBLIC GrB_Semiring LAGraph_any_one_fp64   = NULL ;

//------------------------------------------------------------------------------
// LAGr_Init
//------------------------------------------------------------------------------

LAGRAPH_PUBLIC
int LAGr_Init
(
    // input:
    GrB_Mode mode,      // mode for GrB_Init or GxB_Init
    void * (* user_malloc_function  ) (size_t),
    void * (* user_calloc_function  ) (size_t, size_t),
    void * (* user_realloc_function ) (void *, size_t),
    void   (* user_free_function    ) (void *),
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // malloc and free are required; calloc and realloc are optional
    LG_CLEAR_MSG ;
    LG_ASSERT (user_malloc_function != NULL, GrB_NULL_POINTER) ;
    LG_ASSERT (user_free_function   != NULL, GrB_NULL_POINTER) ;
    GrB_Info info ;

    // ensure LAGr_Init has not already been called
    LG_ASSERT_MSG (!LG_get_LAGr_Init_has_been_called ( ), GrB_INVALID_VALUE,
        "LAGr*_Init can only be called once") ;

    //--------------------------------------------------------------------------
    // start GraphBLAS
    //--------------------------------------------------------------------------

    #if LAGRAPH_SUITESPARSE

        info = GxB_init (mode,
            user_malloc_function,
            user_calloc_function,
            user_realloc_function,
            user_free_function) ;

    #else

        // GxB_init is not available.  Use GrB_init instead.
        info = GrB_init (mode) ;

    #endif

    LG_ASSERT_MSG (info == GrB_SUCCESS || info == GrB_INVALID_VALUE, info,
        "failed to initialize GraphBLAS") ;

    #undef  LG_FREE_ALL
    #define LG_FREE_ALL             \
    {                               \
        LAGraph_Finalize (msg) ;    \
    }

    //--------------------------------------------------------------------------
    // save the memory management pointers in global LAGraph space
    //--------------------------------------------------------------------------

    LAGraph_Malloc_function  = user_malloc_function ;
    LAGraph_Calloc_function  = user_calloc_function ;
    LAGraph_Realloc_function = user_realloc_function ;
    LAGraph_Free_function    = user_free_function ;

    //--------------------------------------------------------------------------
    // set # of LAGraph threads
    //--------------------------------------------------------------------------

    LG_nthreads_outer = 1 ;             // for LAGraph itself, if nested
                                        // regions call GraphBLAS
    #ifdef _OPENMP
    LG_nthreads_inner = omp_get_max_threads ( ) ; // for lower-level parallelism
    #else
    LG_nthreads_inner = 1 ;
    #endif

    #if LAGRAPH_SUITESPARSE
    {
        GRB_TRY (GxB_set (GxB_NTHREADS, LG_nthreads_inner)) ;
    }
    #endif

    //--------------------------------------------------------------------------
    // create global objects
    //--------------------------------------------------------------------------

    // LAGraph_plus_first_T: using the GrB_PLUS_MONOID_T monoid and the
    // GrB_FIRST_T multiplicative operator.  These semirings compute C=A*B
    // where only the structure of B is accessed.  In MATLAB, this can be
    // written as:
    //
    //      C = A * spones (B)

    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_first_int8,
        GrB_PLUS_MONOID_INT8  , GrB_FIRST_INT8  )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_first_int16,
        GrB_PLUS_MONOID_INT16 , GrB_FIRST_INT16 )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_first_int32,
        GrB_PLUS_MONOID_INT32 , GrB_FIRST_INT32 )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_first_int64,
        GrB_PLUS_MONOID_INT64 , GrB_FIRST_INT64 )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_first_uint8,
        GrB_PLUS_MONOID_UINT8 , GrB_FIRST_UINT8 )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_first_uint16,
        GrB_PLUS_MONOID_UINT16, GrB_FIRST_UINT16)) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_first_uint32,
        GrB_PLUS_MONOID_UINT32, GrB_FIRST_UINT32)) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_first_uint64,
        GrB_PLUS_MONOID_UINT64, GrB_FIRST_UINT64)) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_first_fp32,
        GrB_PLUS_MONOID_FP32  , GrB_FIRST_FP32  )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_first_fp64,
        GrB_PLUS_MONOID_FP64  , GrB_FIRST_FP64  )) ;

    // LAGraph_plus_second_T: using the GrB_PLUS_MONOID_T monoid and the
    // GrB_SECOND_T multiplicative operator.  These semirings compute C=A*B
    // where only the structure of A is accessed.  In MATLAB, this can be
    // written as:
    //
    //      C = spones (A) * B

    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_second_int8,
        GrB_PLUS_MONOID_INT8  , GrB_SECOND_INT8  )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_second_int16,
        GrB_PLUS_MONOID_INT16 , GrB_SECOND_INT16 )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_second_int32,
        GrB_PLUS_MONOID_INT32 , GrB_SECOND_INT32 )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_second_int64,
        GrB_PLUS_MONOID_INT64 , GrB_SECOND_INT64 )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_second_uint8,
        GrB_PLUS_MONOID_UINT8 , GrB_SECOND_UINT8 )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_second_uint16,
        GrB_PLUS_MONOID_UINT16, GrB_SECOND_UINT16)) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_second_uint32,
        GrB_PLUS_MONOID_UINT32, GrB_SECOND_UINT32)) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_second_uint64,
        GrB_PLUS_MONOID_UINT64, GrB_SECOND_UINT64)) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_second_fp32,
        GrB_PLUS_MONOID_FP32  , GrB_SECOND_FP32  )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_second_fp64,
        GrB_PLUS_MONOID_FP64  , GrB_SECOND_FP64  )) ;

    // LAGraph_plus_one_T: using the GrB_PLUS_MONOID_T monoid and the
    // corresponding GrB_ONEB_T multiplicative operator.  These semirings
    // compute a matrix C=A*B that does not depend on the type or values of
    // the matrices A and B.  C(i,j) is the size of the intersection of the
    // structures of A(i,:) and B(:,j).  In MATLAB, for the FP64 data type,
    // this can be written as:
    //
    //      C = spones (A) * spones (B)

    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_one_int8,
        GrB_PLUS_MONOID_INT8  , GrB_ONEB_INT8  )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_one_int16,
        GrB_PLUS_MONOID_INT16 , GrB_ONEB_INT16 )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_one_int32,
        GrB_PLUS_MONOID_INT32 , GrB_ONEB_INT32 )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_one_int64,
        GrB_PLUS_MONOID_INT64 , GrB_ONEB_INT64 )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_one_uint8,
        GrB_PLUS_MONOID_UINT8 , GrB_ONEB_UINT8 )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_one_uint16,
        GrB_PLUS_MONOID_UINT16, GrB_ONEB_UINT16)) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_one_uint32,
        GrB_PLUS_MONOID_UINT32, GrB_ONEB_UINT32)) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_one_uint64,
        GrB_PLUS_MONOID_UINT64, GrB_ONEB_UINT64)) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_one_fp32,
        GrB_PLUS_MONOID_FP32  , GrB_ONEB_FP32  )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_plus_one_fp64,
        GrB_PLUS_MONOID_FP64  , GrB_ONEB_FP64  )) ;

    // LAGraph_any_one_T: using the GrB_MIN_MONOID_T for non-boolean types,
    // or GrB_LOR_MONOID_BOOL for boolean, and the GrB_ONEB_T multiplicative
    // operator.  Given any matrices A and B, C = A*B when using this semiring
    // computes a matrix C whose values (for entries present) are all equal to
    // 1.  The result is dependent only on the structure of A and B, not their
    // data types or values.  In MATLAB, this could be written for FP64 as:
    //
    //      C = spones (spones (A) * spones (B))
    //
    // The MIN monoid could also be MAX, TIMES, or GxB_ANY (for SuiteSparse
    // GraphBLAS), or it could be BOR or BAND for the unsigned integer types.
    // The LOR monoid could also be LAND or EQ.  All of these monoids reduce
    // a set of values { 1, 1, 1, ... 1, 1 } down to the single scalar value
    // of 1, or true, and thus any of these monoids will compute the same
    // thing.

    GRB_TRY (GrB_Semiring_new (&LAGraph_any_one_bool,
        GrB_LOR_MONOID_BOOL   , GrB_ONEB_BOOL  )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_any_one_int8,
        GrB_MIN_MONOID_INT8   , GrB_ONEB_INT8  )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_any_one_int16,
        GrB_MIN_MONOID_INT16  , GrB_ONEB_INT16 )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_any_one_int32,
        GrB_MIN_MONOID_INT32  , GrB_ONEB_INT32 )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_any_one_int64,
        GrB_MIN_MONOID_INT64  , GrB_ONEB_INT64 )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_any_one_uint8,
        GrB_MIN_MONOID_UINT8  , GrB_ONEB_UINT8 )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_any_one_uint16,
        GrB_MIN_MONOID_UINT16 , GrB_ONEB_UINT16)) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_any_one_uint32,
        GrB_MIN_MONOID_UINT32 , GrB_ONEB_UINT32)) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_any_one_uint64,
        GrB_MIN_MONOID_UINT64 , GrB_ONEB_UINT64)) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_any_one_fp32,
        GrB_MIN_MONOID_FP32   , GrB_ONEB_FP32  )) ;
    GRB_TRY (GrB_Semiring_new (&LAGraph_any_one_fp64,
        GrB_MIN_MONOID_FP64   , GrB_ONEB_FP64  )) ;

    LG_set_LAGr_Init_has_been_called (true) ;
    return (GrB_SUCCESS) ;
}

