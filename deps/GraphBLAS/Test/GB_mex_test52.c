//------------------------------------------------------------------------------
// GB_mex_test52: test reshape with large matrices
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Triggers a bug in v10.5.1; fixed in v10.5.2 (where T=A' must swap
// i_is_32 and j_is_32, when transposing the input matrix).

#include "GB_mex.h"
#include "GB_mex_errors.h"

#undef  FREE_ALL
#define FREE_ALL                    \
{                                   \
    GrB_Matrix_free (&A) ;          \
    GrB_Matrix_free (&B) ;          \
    GrB_Matrix_free (&C) ;          \
}

//------------------------------------------------------------------------------
// GB_mex_test51 mexFunction
//------------------------------------------------------------------------------

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    GrB_Info info ;
    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL, B = NULL, C = NULL ;

    //--------------------------------------------------------------------------
    // test reshape
    //--------------------------------------------------------------------------

    GrB_Index n1 =  268435456L ;     // 2^28
    GrB_Index n2 = 4294967296L ;     // 2^32
    GrB_Index m2 =   16777216L ;     // 2^24

    // create a matrix of size n1-by-n1 with 1000 entries, type GrB_FP64,
    // held by row
    simple_rand_seed (1) ;
    OK (GB_mx_random_matrix (&A, false, false, n1, n1, 1000, 1, false)) ;
    OK (GrB_Matrix_set_INT32 (A, GrB_ROWMAJOR, GrB_STORAGE_ORIENTATION_HINT)) ;
    // OK (GxB_Matrix_fprint (A, "A", 2, NULL)) ;

    // B = reshape (A, m2, n2), by column
    OK (GxB_Matrix_reshapeDup (&B, A, /* by col: */ true, m2, n2, NULL)) ;
    // OK (GxB_Matrix_fprint (B, "B", 2, NULL)) ;

    // C = reshape (B, n1, n1), by column
    OK (GxB_Matrix_reshapeDup (&C, B, /* by col: */ true, n1, n1, NULL)) ;
    // OK (GxB_Matrix_fprint (C, "C", 2, NULL)) ;

    // assert that A == C
    bool ok = GB_mx_isequal (A, C, 0) ;
    CHECK (ok) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    FREE_ALL ;
    GB_mx_put_global (true) ;
    printf ("GB_mex_test52:  all tests passed\n") ;
}

