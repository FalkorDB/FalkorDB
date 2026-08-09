//------------------------------------------------------------------------------
// gb_get_deep: create a deep GrB_Matrix from a MATLAB input matrix/object
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Returns a deep copy GrB_Matrix C for a GrB or MATLAB input matrix, with no
// pending work.  Used for methods such as
//
//      C = GrB.apply (Cin, ... )
//
// where Cin is either a MATLAB matrix, or a GrB object that must not be
// modified (except any pending work is finished in Cin).  The caller does not
// need Cin, just its deep copy C, which the caller will then modify and
// return as pargout [0].  Thus Cin is not returned to the caller.

// This method is also used for the in-place syntax:
//
//      GrB.apply (C, ... )
//
// with nargout = 0, since in this case, C is modified in place (and it must
// also be a GrB object, not a MATLAB matrix).  In this case, C and Cin are
// the same matrix.

#undef  FREE_WORK
#define FREE_WORK                   \
    GrB_Matrix_free (&C_to_free) ;

#undef  FREE_ALL
#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (&C) ;

GrB_Info gb_get_deep        // get the input/output matrix C
(
    // output:
    GrB_Matrix *C_handle,   // matrix C: deep copy if in-place
    // input:
    bool inplace,           // if true, C is modified in-place (C is Cin)
    gb_matrix matrix,       // input MATLAB, GrB, or GhB matrix
    const int arena,
    char err [ERRLEN]
)
{

    //--------------------------------------------------------------------------
    // get the GrB_Matrix Cin and optional C_to_free of a MATLAB matrix
    //--------------------------------------------------------------------------

    GrB_Matrix Cin = NULL, C = NULL, C_to_free = NULL ;
    OK (gb_get_matrix (&Cin, &C_to_free, matrix, arena, err)) ;

    //--------------------------------------------------------------------------
    // get the GrB_Matrix C
    //--------------------------------------------------------------------------

    if (inplace)
    { 

        //----------------------------------------------------------------------
        // usage: GhB.method (C, ...)
        //----------------------------------------------------------------------

        // ensure C is a GhB handle matrix argument
        if (matrix->G == NULL)
        {
            ERROR ("For in-place syntax, C must be a GhB handle matrix",
                GrB_INVALID_VALUE) ;
        }

        // C is modified in-place.  Any pending work is left undone.
        C = Cin ;

    }
    else
    {

        //----------------------------------------------------------------------
        // usage: C = [GrB,GhB].method (Cin, ...)
        //----------------------------------------------------------------------

        if (matrix->will_wait)
        { 
            // ensure Cin has no pending work
            OK (GrB_Matrix_wait (Cin, GrB_MATERIALIZE)) ;
        }

        // make a deep copy of Cin
        OK (gb_dup (&C, Cin, arena, err)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    // sanity check
    int readonly ;
    OK (GrB_Matrix_get_INT32 (C, &readonly, GxB_IS_READONLY)) ;
    CHECK_ERROR (readonly, "matrix cannot have read-only content") ;

    FREE_WORK ;
    (*C_handle) = C ;
    return (GrB_SUCCESS) ;
}

#undef  FREE_WORK
#define FREE_WORK
#undef  FREE_ALL
#define FREE_ALL

