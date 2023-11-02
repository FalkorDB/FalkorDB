//------------------------------------------------------------------------------
// GB_Scalar_reduce: reduce a matrix to a GrB_Scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_reduce.h"

#define GB_FREE_ALL GB_phybix_free ((GrB_Matrix) S) ;

GrB_Info GB_Scalar_reduce
(
    GrB_Scalar S,                   // result scalar
    const GrB_BinaryOp accum,       // optional accum for c=accum(c,t)
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A,             // matrix to reduce
    GB_Werk Werk
)
{ 

    //--------------------------------------------------------------------------
    // check inputs (accum and monoid are checked in GB_reduce_to_scalar)
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_SCALAR_OK (S, "S for reduce to GrB_Scalar", GB0) ;
    ASSERT_MATRIX_OK (A, "A for reduce to GrB_Scalar", GB0) ;
    GBURBLE ("(to GrB_Scalar) ") ;

    //--------------------------------------------------------------------------
    // check if A is empty
    //--------------------------------------------------------------------------

    GrB_Index nvals ;
    GB_OK (GB_nvals (&nvals, A, Werk)) ;
    if (nvals == 0)
    {
        // no work to do, except to clear S if there is no accum operator
        if (accum == NULL)
        { 
            GB_OK (GB_clear ((GrB_Matrix) S, Werk)) ;
        }
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // ensure S is bitmap or full
    //--------------------------------------------------------------------------

    if (GB_IS_FULL (S))
    { 
        // nothing to do
        nvals = 1 ;
    }
    else
    { 
        // convert S to bitmap
        GB_OK (GB_convert_any_to_bitmap ((GrB_Matrix) S, Werk)) ;
        nvals = S->nvals ;
    }

    //--------------------------------------------------------------------------
    // reduce to the GrB_Scalar
    //--------------------------------------------------------------------------

    // ignore accum if S has no entry on input
    GrB_BinaryOp accum_op = (nvals == 0) ? NULL : accum ;
    GB_OK (GB_reduce_to_scalar (S->x, S->type, accum_op, monoid, A, Werk)) ;

    //--------------------------------------------------------------------------
    // ensure S is full
    //--------------------------------------------------------------------------

    if (GB_IS_BITMAP (S))
    { 
        S->b [0] = 1 ;
        S->nvals = 1 ;
        GB_convert_any_to_full ((GrB_Matrix) S) ;
    }
    S->iso = true ;
    ASSERT_SCALAR_OK (S, "S result for reduce to GrB_Scalar", GB0) ;
    return (GrB_SUCCESS) ;
}

