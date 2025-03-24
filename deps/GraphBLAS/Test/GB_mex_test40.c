//------------------------------------------------------------------------------
// GB_mex_test40: GB_ix_realloc
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#undef  FREE_ALL
#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free (&A) ;              \
}

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
    GrB_Matrix A = NULL ;

    //--------------------------------------------------------------------------
    // test GB_ix_realloc
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&A, GrB_BOOL, 10, 10)) ;
    OK (GrB_Matrix_set_INT32 (A, 32, GxB_ROWINDEX_INTEGER_HINT)) ;
    OK (GrB_Matrix_set_INT32 (A, 32, GxB_COLINDEX_INTEGER_HINT)) ;
    OK (GrB_Matrix_set_INT32 (A, 32, GxB_OFFSET_INTEGER_HINT)) ;
    OK (GrB_Matrix_setElement (A, 1, 1, 1)) ;
    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
    OK (GxB_Matrix_fprint (A, "A for ix_realloc", 5, NULL)) ;
    int64_t nzmax_new = ((int64_t) UINT32_MAX) + 10 ;
    printf ("nzmax_new %ld\n", nzmax_new) ;
    OK (GB_ix_realloc (A, nzmax_new)) ;
    OK (GxB_Matrix_fprint (A, "A for ix_realloc; realloced", 5, NULL)) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    FREE_ALL ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test40:  all tests passed\n\n") ;
}

