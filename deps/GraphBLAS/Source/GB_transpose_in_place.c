//------------------------------------------------------------------------------
// GB_transpose_in_place: in-place transpose
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// All other uses of GB_transpose are not in-place.
// No operator is applied and no typecasting is done.

#include "GB_transpose.h"

GrB_Info GB_transpose_in_place   // C=A', no change of type, no operators
(
    GrB_Matrix C,               // output matrix C, possibly modified in-place
    const bool C_is_csc,        // desired CSR/CSC format of C
    GB_Werk Werk
)
{ 
    return (GB_transpose (C, NULL, C_is_csc, C,
        NULL, NULL, false, false,       // no operator
        Werk)) ;
}

