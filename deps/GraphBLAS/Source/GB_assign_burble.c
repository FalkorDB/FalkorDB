//------------------------------------------------------------------------------
// GB_assign_burble.c: burble the assign method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

void GB_assign_burble
(
    const bool C_replace,       // descriptor for C
    const int Ikind,
    const int Jkind,
    const GrB_Matrix M,         // mask matrix, which is not NULL here
    const bool Mask_comp,       // true for !M, false for M
    const bool Mask_struct,     // true if M is structural, false if valued
    const GrB_BinaryOp accum,   // present here
    const GrB_Matrix A,         // input matrix, not transposed
    const int assign_kind       // row assign, col assign, assign, or subassign
)
{ 

    //--------------------------------------------------------------------------
    // quick return if burble is disabled
    //--------------------------------------------------------------------------

    if (!GB_Global_burble_get ( ))
    { 
        return ;
    }

    //--------------------------------------------------------------------------
    // construct the string that describes the method
    //--------------------------------------------------------------------------

    #define SLEN 512
    char description [SLEN+1] ;
    GB_assign_describe (description, SLEN, C_replace, Ikind, Jkind,
        M == NULL, GB_sparsity (M), Mask_comp, Mask_struct,
        accum, A == NULL, assign_kind) ;

    //--------------------------------------------------------------------------
    // burble the description
    //--------------------------------------------------------------------------

    GBURBLE ("%s", description) ;
}

