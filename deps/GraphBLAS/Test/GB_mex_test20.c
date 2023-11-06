//------------------------------------------------------------------------------
// GB_mex_test20: test GB_mask
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"
#include "GB_mask.h"

#define USAGE "GB_mex_test20"

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
    GB_WERK ("GB_mex_test20") ;

    //--------------------------------------------------------------------------
    // test GB_masker
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, Z = NULL ;
    int n = 5 ;
    GxB_set (GxB_BURBLE, true) ;

    for (int trial = 1 ; trial <= 2 ; trial++)
    {
        OK (GrB_Matrix_new (&C, GrB_FP32, n, n)) ;
        OK (GrB_Matrix_new (&Z, GrB_FP32, n, n)) ;
        OK (GxB_set (C, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
        OK (GxB_set (Z, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
        OK (GxB_print (Z, 3)) ;
        for (int k = 0 ; k < n ; k++)
        {
            OK (GrB_Matrix_setElement (C, ((trial == 1) ? 1 : k), k, (k+1)%n)) ;
        }
        OK (GrB_wait (C, GrB_MATERIALIZE)) ;
        OK (GxB_print (C, 3)) ;
        bool Mask_struct = C->iso ;
        OK (GB_mask (C, C, &Z, false, false, Mask_struct, Werk)) ;
        OK (GxB_print (C, 3)) ;
        GrB_free (&C) ;
    }

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GxB_set (GxB_BURBLE, false) ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test20:  all tests passed\n\n") ;
}

