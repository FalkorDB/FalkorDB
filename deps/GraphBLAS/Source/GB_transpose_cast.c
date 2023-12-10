//------------------------------------------------------------------------------
// GB_transpose_cast: transpose and typecast
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The transpose is not in-place.  No operator is applied.  C = (ctype) A' is
// computed, or C = (ctype) one (A') if iso_one is true, with typecasting if
// ctype is not equal to A->type.  If iso_one is true, C is returned as an
// iso matrix, with an iso value of 1.

#include "GB_transpose.h"
#include "GB_unop.h"

GrB_Info GB_transpose_cast      // C= (ctype) A' or one (A'), not in-place
(
    GrB_Matrix C,               // output matrix C, not in place
    GrB_Type ctype,             // desired type of C
    const bool C_is_csc,        // desired CSR/CSC format of C
    const GrB_Matrix A,         // input matrix; C != A
    const bool iso_one,         // if true, C = one (A'), as iso
    GB_Werk Werk
)
{ 
    ASSERT (C != A && !GB_any_aliased (C, A)) ;

    GB_Operator op = (GB_Operator)
        ((iso_one) ? GB_unop_one (ctype->code) : NULL) ;

    // C = (ctype) A' if op is NULL, or C = (ctype) one (A')
    return (GB_transpose (C, ctype, C_is_csc, A,
        op, NULL, false, false,     // iso ONE operator or NULL
        Werk)) ;
}

