//------------------------------------------------------------------------------
// gb_get_sparsity: determine the sparsity of a matrix result 
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
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

// This method does not allocate any memory, so it is safe to use in either
// the GrB* or mx* region of a mexFunction.

GrB_Info gb_get_sparsity    // determine the sparsity of C for C = method(A,B)
(
    // input:
    GrB_Matrix A,           // may be NULL
    GrB_Matrix B,           // may be NULL
    // input/output:
    int *sparsity,          // may be 0 on input
    char err [ERRLEN]
)
{

    int A_sparsity = 0 ;
    int B_sparsity = 0 ;
    uint64_t nrows, ncols ;

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
            OK (GrB_Matrix_get_INT32 (A, &A_sparsity, GxB_SPARSITY_CONTROL)) ;
        }
    }

    if (B != NULL)
    { 
        OK (GrB_Matrix_nrows (&nrows, B)) ;
        OK (GrB_Matrix_ncols (&ncols, B)) ;
        if (nrows > 1 || ncols > 1)
        { 
            // B is a vector or matrix, not a scalar
            OK (GrB_Matrix_get_INT32 (B, &B_sparsity, GxB_SPARSITY_CONTROL)) ;
        }
    }

    //--------------------------------------------------------------------------
    // determine the sparsity of C
    //--------------------------------------------------------------------------

    if ((*sparsity) != 0)
    { 
        // (2) the sparsity is defined by the descriptor to the method
    }
    else if (A_sparsity > 0 && B_sparsity > 0)
    { 
        // (3) C is determined by the sparsity of A and B
        (*sparsity) = A_sparsity | B_sparsity ;
    }
    else if (A_sparsity > 0)
    { 
        // (4) get the sparsity of A
        (*sparsity) = A_sparsity ;
    }
    else if (B_sparsity > 0)
    { 
        // (5) get the sparsity of B
        (*sparsity) = B_sparsity ;
    }
    else
    { 
        // (6) use the default sparsity
        (*sparsity) = GxB_AUTO_SPARSITY ;
    }

    return (GrB_SUCCESS) ;
}

