//------------------------------------------------------------------------------
// gbmex_assign: assign entries into a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmex_assign is an interface to GrB_Matrix_assign and
// GrB_Matrix_assign_[TYPE], for implementing GrB.assign and GhB.assign.

// Usage for GrB and GhB (omitting desc argument):

// C = GrB.assign (Cin, A, I, J)                C = Cin ; C(I,J) = A
// C = GrB.assign (Cin, accum, A, I, J)         C = Cin ; C(I,J) += A
// C = GrB.assign (Cin, M, A, I, J)             C = Cin ; C<M>(I,J) = A
// C = GrB.assign (Cin, M, accum, A, I, J)      C = Cin ; C<M>(I,J) += A

// Usage for GhB only (inplace):

// GhB.assign (C, A, I, J)                      C(I,J) = A
// GhB.assign (C, accum, A, I, J)               C(I,J) += A
// GhB.assign (C, M, A, I, J)                   C<M>(I,J) = A
// GhB.assign (C, M, accum, A, I, J)            C<M>(I,J) += A

// A can be a matrix or a scalar.  I and J are optional.

#include "gb_interface.h"
#include "gb_cell_to_list.c"
#include "gb_matrix_to_list.c"
#include "gb_string_and_type_to_binop_or_idxunop.c"
#include "gb_string_to_binop.c"
#include "gb_string_to_binop_or_idxunop.c"

#include "gbmx_interface.h"
#include "gbmx_assign_mexFunction.c"

#define USAGE "usage: C = GrB.assign (Cin, M, accum, A, I, J, desc)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    gbmx_assign_mexFunction (nargout, pargout, nargin, pargin, false, USAGE) ;
}

