//------------------------------------------------------------------------------
// gb_get_sparsity: determine the sparsity of a matrix result 
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gb_get_sparsity determines the sparsity of a result matrix C, which may be
// computed from one or two input matrices A and B.  The following rules are
// used, in order:

// (1) GraphBLAS operations of the form C = GrB.method (Cin, ...) use the
//      sparsity of Cin for the new matrix C.

// (2) If the sparsity is determined by the descriptor to the method, then that
//      determines the sparsity of C.

// (3) If both A and B are present and both matrices (not scalars), the
//      sparsity of C is A_sparsity | B_sparsity

// (4) If A is present (and not a scalar), then the sparsity of C is A_sparsity.

// (5) If B is present (and not a scalar), then the sparsity of C is B_sparsity.

// (6) Otherwise, the global default sparsity is used for C.

#include "gb_interface.h"

GxB_Format_Value gb_get_sparsity        // 0 to 15
(
    GrB_Matrix A,                       // may be NULL
    GrB_Matrix B,                       // may be NULL
    int sparsity_default                // may be 0
)
{

    int sparsity ;
    int A_sparsity = 0 ;
    int B_sparsity = 0 ;
    GrB_Index nrows, ncols ;

    //--------------------------------------------------------------------------
    // get the sparsity of the matrices A and B
    //--------------------------------------------------------------------------

    if (A != NULL)
    {
        OK (GrB_Matrix_nrows (&nrows, A)) ;
        OK (GrB_Matrix_ncols (&ncols, A)) ;
        if (nrows > 1 || ncols > 1)
        {
            // A is a vector or matrix, not a scalar
            OK (GxB_Matrix_Option_get (A, GxB_SPARSITY_CONTROL, &A_sparsity)) ;
        }
    }

    if (B != NULL)
    {
        OK (GrB_Matrix_nrows (&nrows, B)) ;
        OK (GrB_Matrix_ncols (&ncols, B)) ;
        if (nrows > 1 || ncols > 1)
        {
            // B is a vector or matrix, not a scalar
            OK (GxB_Matrix_Option_get (B, GxB_SPARSITY_CONTROL, &B_sparsity)) ;
        }
    }

    //--------------------------------------------------------------------------
    // determine the sparsity of C
    //--------------------------------------------------------------------------

    if (sparsity_default != 0)
    { 
        // (2) the sparsity is defined by the descriptor to the method
        sparsity = sparsity_default ;
    }
    else if (A_sparsity > 0 && B_sparsity > 0)
    {
        // (3) C is determined by the sparsity of A and B
        sparsity = A_sparsity | B_sparsity ;
    }
    else if (A_sparsity > 0)
    {
        // (4) get the sparsity of A
        sparsity = A_sparsity ;
    }
    else if (B_sparsity > 0)
    {
        // (5) get the sparsity of B
        sparsity = B_sparsity ;
    }
    else
    {
        // (6) use the default sparsity
        sparsity = GxB_AUTO_SPARSITY ;
    }

    return (sparsity) ;
}

