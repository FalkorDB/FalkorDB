//------------------------------------------------------------------------------
// GB_ewise_fulln_template: C = A+B where all 3 matrices are full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

//  This template is not used for any generic kernels.

#include "GB_unused.h"

{

    //--------------------------------------------------------------------------
    // get A, B, and C
    //--------------------------------------------------------------------------

    // any matrix may be aliased to any other (C==A, C==B, and/or A==B)
    GB_A_TYPE *Ax = (GB_A_TYPE *) A->x ;
    GB_B_TYPE *Bx = (GB_B_TYPE *) B->x ;
    GB_C_TYPE *Cx = (GB_C_TYPE *) C->x ;
    GB_C_NVALS (cnz) ;          // const int64_t cnz = GB_nnz (C) ;
    ASSERT (GB_IS_FULL (A)) ;
    ASSERT (GB_IS_FULL (B)) ;
    ASSERT (GB_IS_FULL (C)) ;
    ASSERT (!C->iso) ;
    ASSERT (!A->iso) ;
    ASSERT (!B->iso) ;
    int64_t p ;

    //--------------------------------------------------------------------------
    // C = A+B where all 3 matrices are full
    //--------------------------------------------------------------------------

    // note that A, B and C may all be aliased to each other, in any way
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (p = 0 ; p < cnz ; p++)
    { 
        GB_DECLAREA (aij) ;
        GB_GETA (aij, Ax, p, false) ;               // aij = Ax [p]
        GB_DECLAREB (bij) ;
        GB_GETB (bij, Bx, p, false) ;               // bij = Bx [p]
        GB_EWISEOP (Cx, p, aij, bij, 0, 0) ;        // Cx [p] = aij + bij
    }
}

