//------------------------------------------------------------------------------
// gbmex_eadd: sparse matrix addition
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"
#include "gb_string_and_type_to_binop_or_idxunop.c"
#include "gb_string_to_binop.c"
#include "gb_string_to_binop_or_idxunop.c"

#include "gbmx_interface.h"

#include "gbmx_ewise_mexFunction.c"

#define USAGE "usage: C = GrB.eadd (Cin, M, accum, binop, A, B, desc)"

// Usage for GrB and GhB (omitting desc argument):

// C = GrB.eadd (op, A, B)                 C = op(A,B)
// C = GrB.eadd (Cin, op, A, B)            C = op(A,B)
// C = GrB.eadd (Cin, accum, op, A, B)     C = Cin + op(A,B)
// C = GrB.eadd (Cin, M, op, A, B)         C = Cin ; C<M> = op(A,B)
// C = GrB.eadd (Cin, M, accum, op, A, B)  C = Cin ; C<M> += op(A,B)

// Usage for GhB only (inplace):

// GhB.eadd (C, op, A, B)                  C = op(A,B)
// GhB.eadd (C, accum, op, A, B)           C += op(A,B)
// GhB.eadd (C, M, op, A, B)               C<M> = op(A,B)
// GhB.eadd (C, M, accum, op, A, B)        C<M> += op(A,B)

// where op(A,B) refers to eWiseAdd, A+B, using the given op.

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    gbmx_ewise_mexFunction (nargout, pargout, nargin, pargin, true, USAGE) ;
}

