//------------------------------------------------------------------------------
// gbmex_extracttuples: extract all entries from a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// [I J X] = GrB.extracttuples (A)
// [I J X] = GrB.extracttuples (A, desc)

// The desciptor is optional.  If present, it must be a struct.

// desc.base = 'zero-based':    I and J are returned as 0-based integer indices
// desc.base = 'one-based int': I and J are returned as 1-based integer indices
// desc.base = 'one-based':     I and J are returned as 1-based integer indices
// desc.base = 'one-based double' one-based double unless max(size(A)) >
//                              flintmax, in which case 'one-based int' is used.
// desc.base = 'default':       'one-based int'

// The input matrix must have no pending work.

// I, J, and X are returned as built-in MATLAB/Octave matrices.

// FUTURE: add an option to return I,J,X as GrB matrices instead
// FUTURE: reduce # of copies made

#include "gb_interface.h"
#include "gbmx_interface.h"

#undef  FREE_ALL
#define FREE_ALL                    \
    GrB_Vector_free (&I) ;          \
    GrB_Vector_free (&J) ;          \
    GrB_Vector_free (&X) ;          \
    GrB_Vector_free (&T) ;          \
    GrB_Matrix_free (&A_to_free) ;  \
    gb_free (&x, xarena) ;

#define USAGE "usage: [I,J,X] = GrB.extracttuples (A, desc)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs and construct outputs
    //--------------------------------------------------------------------------

    GrB_Matrix A = NULL, A_to_free = NULL ;
    GrB_Vector I = NULL, J = NULL, X = NULL, T = NULL ;
    void *x = NULL ;

    GBMX_USAGE (nargin >= 2 && nargin <= 3 && nargout <= 3, USAGE) ;
    bool ghb = (bool) mxGetScalar (pargin [0]) ;
    int arena = ghb ? GrB_DEFAULT : MXARENA ;
    int xarena = GrB_DEFAULT ;      // revised below

    //--------------------------------------------------------------------------
    // find the arguments
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [6] ;
    mxArray *Cell [2] ;
    char String [2][LEN+2] ;
    int nmatrices, nstrings, ncells ;
    struct gb_descriptor_struct gbdesc ;
    gbmx_get_mxargs (nargin, pargin, USAGE, Matrix, &nmatrices, String,
        &nstrings, Cell, &ncells, &gbdesc) ;

    CHECK_ERROR (nmatrices != 1 || nstrings > 0 || ncells > 0, USAGE) ;

    CHECK_ERROR (Matrix [0].will_wait, "matrix must have no pending work") ;

    //--------------------------------------------------------------------------
    // construct I, J, X outputs
    //--------------------------------------------------------------------------

    int64_t nvals = Matrix [0].nvals ;
    int64_t nrows = Matrix [0].nrows ;
    int64_t ncols = Matrix [0].ncols ;
    GrB_Type X_type = Matrix [0].type ;
    size_t X_typesize = Matrix [0].typesize ;

    bool extract_I = true ;
    bool extract_J = (nargout > 1) ;
    bool extract_X = (nargout > 2) ;

    if (gbdesc.base == BASE_1_DOUBLE && MAX (nrows, ncols) > FLINTMAX)
    { 
        gbdesc.base = BASE_1_INT ;
    }

    GrB_Type I_type, J_type ;
    if (gbdesc.base == BASE_1_DOUBLE)
    { 
        I_type = GrB_FP64 ;
        J_type = GrB_FP64 ;
    }
    else
    { 
        bool I_is_32 = (nrows <= INT32_MAX) ;
        bool J_is_32 = (ncols <= INT32_MAX) ;
        I_type = (I_is_32) ? GrB_INT32 : GrB_INT64 ;
        J_type = (J_is_32) ? GrB_INT32 : GrB_INT64 ;
    }

    void *I_out = NULL, *J_out = NULL, *X_out = NULL ;
    size_t I_typesize, J_typesize ;
    OK (GxB_Type_size (&I_typesize, I_type)) ;
    OK (GxB_Type_size (&J_typesize, J_type)) ;

    if (extract_I)
    { 
        pargout [0] = gbmx_new_matlab_matrix (nvals, 1, I_type) ;
        I_out = mxGetData (pargout [0]) ;
    }
    if (extract_J)
    { 
        pargout [1] = gbmx_new_matlab_matrix (nvals, 1, J_type) ;
        J_out = mxGetData (pargout [1]) ;
    }
    if (extract_X)
    { 
        pargout [2] = gbmx_new_matlab_matrix (nvals, 1, X_type) ;
        X_out = mxGetData (pargout [2]) ;
    }

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the # of threads to use
    //--------------------------------------------------------------------------

    int nthreads ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &nthreads, GxB_NTHREADS)) ;

    //--------------------------------------------------------------------------
    // get the matrix; disable burble for scalars
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;
    int burble ;
    bool disable_burble = (nrows <= 1 && ncols <= 1) ;
    if (disable_burble)
    { 
        OK (GrB_Global_get_INT32 (GrB_GLOBAL, &burble, GxB_BURBLE)) ;
        OK (GrB_Global_set_INT32 (GrB_GLOBAL, false, GxB_BURBLE)) ;
    }

    //--------------------------------------------------------------------------
    // create empty GrB_Vectors for I, J, and X
    //--------------------------------------------------------------------------

    // type of I and J will be revised as needed by GxB_Matrix_extractTuples
    if (extract_I) OK (GxB_Vector_new_arena (&I, GrB_UINT64, 0, arena, arena)) ;
    if (extract_J) OK (GxB_Vector_new_arena (&J, GrB_UINT64, 0, arena, arena)) ;
    if (extract_X) OK (GxB_Vector_new_arena (&X, X_type, 0, arena, arena)) ;

    //--------------------------------------------------------------------------
    // extract the tuples from A into I, J, and X
    //--------------------------------------------------------------------------

    OK (GxB_Matrix_extractTuples_Vector (I, J, X, A, NULL)) ;

    //--------------------------------------------------------------------------
    // determine if 1 must be added to the indices
    //--------------------------------------------------------------------------

    int base_offset = (gbdesc.base == BASE_0_INT) ? 0 : 1 ;

    //--------------------------------------------------------------------------
    // return I to MATLAB
    //--------------------------------------------------------------------------

    uint64_t size = 0, nvals2 = 0 ;
    int ignore = 0 ;
    GrB_Type type = NULL ;

    if (extract_I)
    { 
        if (gbdesc.base == BASE_1_DOUBLE)
        { 
            // I = (double) (I + 1)
            OK (GxB_Vector_new_arena (&T, GrB_FP64, nvals, arena, arena)) ;
            OK (GrB_Vector_apply_BinaryOp2nd_FP64 (T, NULL, NULL,
                GrB_PLUS_FP64, I, base_offset, NULL)) ;
            GrB_Vector_free (&I) ;
            I = T ;
            T = NULL ;
        }
        else if (base_offset != 0)
        { 
            // I = I+1, as a uint64 or uint32 vector
            OK (GrB_Vector_apply_BinaryOp2nd_UINT64 (I, NULL, NULL,
                GrB_PLUS_UINT64, I, 1, NULL)) ;
        }
        uint64_t nvals2 ;
        int handling ;
        OK (GxB_Vector_unload (I, &x, &type, &nvals2, &size, &handling, NULL)) ;
        xarena = (handling >= GxB_IS_READONLY) ?
            (handling - GxB_IS_READONLY) : handling ;
        if (type == GrB_UINT32) type = GrB_INT32 ;
        if (type == GrB_UINT64) type = GrB_INT64 ;
        ASSERT (type == I_type) ;
        ASSERT (nvals == nvals2) ;
        GB_memcpy (I_out, x, nvals * I_typesize, nthreads) ;
        gb_free (&x, xarena) ;
        GrB_Vector_free (&I) ;
    }

    //--------------------------------------------------------------------------
    // return J to MATLAB
    //--------------------------------------------------------------------------

    if (extract_J)
    { 
        if (gbdesc.base == BASE_1_DOUBLE)
        { 
            // J = (double) (J + 1)
            OK (GxB_Vector_new_arena (&T, GrB_FP64, nvals, arena, arena)) ;
            OK (GrB_Vector_apply_BinaryOp2nd_FP64 (T, NULL, NULL,
                GrB_PLUS_FP64, J, base_offset, NULL)) ;
            GrB_Vector_free (&J) ;
            J = T ;
            T = NULL ;
        }
        else if (base_offset != 0)
        { 
            // J = J+1, as a uint64 or uint32 vector
            OK (GrB_Vector_apply_BinaryOp2nd_UINT64 (J, NULL, NULL,
                GrB_PLUS_UINT64, J, 1, NULL)) ;
        }
        int handling ;
        OK (GxB_Vector_unload (J, &x, &type, &nvals2, &size, &handling, NULL)) ;
        xarena = (handling >= GxB_IS_READONLY) ?
            (handling - GxB_IS_READONLY) : handling ;
        if (type == GrB_UINT32) type = GrB_INT32 ;
        if (type == GrB_UINT64) type = GrB_INT64 ;
        ASSERT (type == J_type) ;
        ASSERT (nvals == nvals2) ;
        GB_memcpy (J_out, x, nvals * J_typesize, nthreads) ;
        gb_free (&x, xarena) ;
        GrB_Vector_free (&J) ;
    }

    //--------------------------------------------------------------------------
    // return X to MATLAB
    //--------------------------------------------------------------------------

    if (extract_X)
    { 
        int handling ;
        OK (GxB_Vector_unload (X, &x, &type, &nvals2, &size, &handling, NULL)) ;
        xarena = (handling >= GxB_IS_READONLY) ?
            (handling - GxB_IS_READONLY) : handling ;
        ASSERT (type == X_type) ;
        ASSERT (nvals == nvals2) ;
        GB_memcpy (X_out, x, nvals * X_typesize, nthreads) ;
        gb_free (&x, xarena) ;
        GrB_Vector_free (&X) ;
    }

    //--------------------------------------------------------------------------
    // restore burble and return result
    //--------------------------------------------------------------------------

    if (disable_burble)
    { 
        OK (GrB_Global_set_INT32 (GrB_GLOBAL, burble, GxB_BURBLE)) ;
    }

    FREE_ALL ;
    gb_wrapup ( ) ;
}

