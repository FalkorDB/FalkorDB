//------------------------------------------------------------------------------
// gbmex_builtin: convert a GrB matrix to a MATLAB/Octave matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input is a GrB matrix A, but in a limited range of formats and sparsity
// structures, to be compatible with MATLAB/Octave built-in sparse/full
// matrices.  The format is by-column only.  It is sparse or full, never
// bitmap or hypersparse.  It has no pending work.  The integers for a sparse
// matrix are all 64-bit.  The matrix is not iso-valued.  If sparse, the matrix
// must be either GrB_BOOL, GrB_FP64, or GxB_FC64.

// This method does not malloc/free any content of a GraphBLAS matrix, so it
// is safe to use mxMalloc and mxCreate* throughout the mexFunction.  If the
// method fails, MATLAB will automatically destroy the output matrix C, and
// will leave the input GrB matrix A unchanged.

// This strategy allows C to be safely created with no memory leaks.  The only
// downside is that this approach requires a copy to be made.  This method is
// used after another mexFunction has created a GrB matrix with KIND_SPARSE,
// KIND_FULL, or KIND_BUILTIN (either sparse or full), in prepartion for this
// mexFunction, which creates the final MATLAB/Octave sparse/full matrix.

// Usage:

// C = gbmex_builtin (A)

#include "gb_interface.h"
#include "gbmx_interface.h"

#undef  FREE_ALL
#define FREE_ALL GrB_Matrix_free (&A_to_free) ;

#define USAGE "usage: C = gbmex_builtin (A)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Matrix A = NULL, A_to_free = NULL ;
    int arena = GrB_DEFAULT ;   // use default arena for temporary workspace

    GBMX_USAGE (nargin == 1 && nargout == 1, USAGE) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix ;
    gbmx_get_matrix (&Matrix, pargin [0]) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get matrix input
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_to_free, &Matrix, arena, err)) ;
    uint64_t *Ap = (uint64_t *) A->p ;
    uint64_t *Ai = (uint64_t *) A->i ;
    void *Ax = A->x ;

    int sparsity_status ;
    OK (GrB_Matrix_get_INT32 (A, &sparsity_status, GxB_SPARSITY_STATUS)) ;

    //--------------------------------------------------------------------------
    // sanity checks
    //--------------------------------------------------------------------------

    // The input GrB_Matrix must be held by column, with all-64-bit integers,
    // no pending work, non-iso, sparse or full (not hypersparse or bitmap),
    // and if sparse it must have a type of GrB_BOOL, GrB_FP64, or GxB_FC64.
    // This allows the contents of the GrB_Matrix to be copied directly into a
    // MATLAB/Octave matrix via memcpy.

    int fmt, bits, will_wait, iso ;

    CHECK_ERROR (!(sparsity_status == GxB_SPARSE
                || sparsity_status == GxB_FULL), "internal error 1") ;

    OK (GrB_Matrix_get_INT32 (A, &fmt, GxB_FORMAT)) ;
    CHECK_ERROR (fmt != GxB_BY_COL, "internal error 2") ;

    if (sparsity_status == GxB_SPARSE)
    {
        OK (GrB_Matrix_get_INT32 (A, &bits, GxB_OFFSET_INTEGER_BITS)) ;
        CHECK_ERROR (bits != 64, "internal error 3") ;

        OK (GrB_Matrix_get_INT32 (A, &bits, GxB_ROWINDEX_INTEGER_BITS)) ;
        CHECK_ERROR (bits != 64, "internal error 4") ;

        CHECK_ERROR (!(Matrix.type == GrB_BOOL || Matrix.type == GrB_FP64 ||
                    Matrix.type == GxB_FC64), "internal error 5") ;
    }

    OK (GrB_Matrix_get_INT32 (A, &will_wait, GxB_WILL_WAIT)) ;
    CHECK_ERROR (will_wait, "internal error 6") ;

    OK (GrB_Matrix_get_INT32 (A, &iso, GxB_ISO)) ;
    CHECK_ERROR (iso, "internal error 7") ;

    //--------------------------------------------------------------------------
    // construct the output MATLAB/Octave matrix
    //--------------------------------------------------------------------------

    int nthreads ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &nthreads, GxB_NTHREADS)) ;

    if (sparsity_status == GxB_SPARSE)
    {
        if (Matrix.type == GrB_BOOL)
        { 
            pargout [0] = mxCreateSparseLogicalMatrix (Matrix.nrows,
                Matrix.ncols, Matrix.nvals+1) ;
        }
        else if (Matrix.type == GrB_FP64)
        { 
            pargout [0] = mxCreateSparse (Matrix.nrows, Matrix.ncols,
                Matrix.nvals+1, mxREAL) ;
        }
        else // if (Matrix.type == GxB_FC64)
        { 
            pargout [0] = mxCreateSparse (Matrix.nrows, Matrix.ncols,
                Matrix.nvals+1, mxCOMPLEX) ;
        }
        uint64_t *Cp = (uint64_t *) mxGetJc (pargout [0]) ;
        uint64_t *Ci = (uint64_t *) mxGetIr (pargout [0]) ;
        GB_memcpy (Cp, Ap, (Matrix.ncols+1) * sizeof (uint64_t), nthreads) ;
        GB_memcpy (Ci, Ai, Matrix.nvals * sizeof (uint64_t), nthreads) ;
    }
    else
    { 
        pargout [0] = gbmx_new_matlab_matrix (Matrix.nrows, Matrix.ncols,
            Matrix.type) ;
    }

    void *Cx = mxGetData (pargout [0]) ;
    GB_memcpy (Cx, Ax, Matrix.nvals * Matrix.typesize, nthreads) ;
    FREE_ALL ;
    gb_wrapup ( ) ;
}

