//------------------------------------------------------------------------------
// GB_mex_test14: more simple tests
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "GB_mex_test14"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

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

    //--------------------------------------------------------------------------
    // GB_flip_binop
    //--------------------------------------------------------------------------

    bool flipxy ;

    flipxy = true ;
    CHECK (GB_flip_binop (GxB_ISLT_BOOL, &flipxy) == GxB_ISGT_BOOL) ;
    CHECK (!flipxy) ;

    flipxy = true ;
    CHECK (GB_flip_binop (GxB_ISGT_BOOL, &flipxy) == GxB_ISLT_BOOL) ;
    CHECK (!flipxy) ;

    flipxy = true ;
    CHECK (GB_flip_binop (GxB_ISLE_BOOL, &flipxy) == GxB_ISGE_BOOL) ;
    CHECK (!flipxy) ;

    flipxy = true ;
    CHECK (GB_flip_binop (GxB_ISGE_BOOL, &flipxy) == GxB_ISLE_BOOL) ;
    CHECK (!flipxy) ;

    flipxy = true ;
    CHECK (GB_flip_binop (GrB_DIV_BOOL, &flipxy) == GxB_RDIV_BOOL) ;
    CHECK (!flipxy) ;

    flipxy = true ;
    CHECK (GB_flip_binop (GxB_RDIV_BOOL, &flipxy) == GrB_DIV_BOOL) ;
    CHECK (!flipxy) ;

    flipxy = true ;
    CHECK (GB_flip_binop (GrB_MINUS_BOOL, &flipxy) == GxB_RMINUS_BOOL) ;
    CHECK (!flipxy) ;

    flipxy = true ;
    CHECK (GB_flip_binop (GxB_RMINUS_BOOL, &flipxy) == GrB_MINUS_BOOL) ;
    CHECK (!flipxy) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test14:  all tests passed\n\n") ;
}

