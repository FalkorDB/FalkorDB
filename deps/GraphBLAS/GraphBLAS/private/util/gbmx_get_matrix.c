//------------------------------------------------------------------------------
// gbmx_get_matrix: get a matrix argument
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmx_get_matrix (matrix,X) gets the contents of a GraphBLAS GrB or GhB
// matrix object, or the properties of a MATLAB matrix (type, dimensions, and
// pointers to p,i,x, etc), and saves them in the gb_matrix struct.

// X must not be NULL, but it can be an empty matrix, as X = [ ].  In this
// case, the gb_matrix will be 0-by-0.

// This method allocates no memory, and thus mx* and GrB_* methods are
// intermingled.

void gbmx_get_matrix
(
    // output
    gb_matrix matrix,       // either a GraphBLAS or MATLAB matrix, statically
                            // allocated (but undefined) on input
    // input
    const mxArray *X        // GrB or GhB object or MATLAB matrix
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    char err [ERRLEN] ;
    err [0] = '\0' ;
    ASSERT (matrix != NULL) ;
    CHECK_ERROR (X == NULL, "matrix is missing") ;
    memset (matrix, 0, sizeof (struct gb_matrix_struct)) ;

    //--------------------------------------------------------------------------
    // construct the gb_matrix
    //--------------------------------------------------------------------------

    bool is_struct = mxIsStruct (X) ;
    int nfields = (is_struct) ? mxGetNumberOfFields (X) : 0 ;
    bool is_grb = mxIsClass (X, "GrB") || (is_struct && nfields > 1) ;
    bool is_ghb = mxIsClass (X, "GhB") || (is_struct && nfields == 1) ;

    if (is_ghb)
    { 

        //----------------------------------------------------------------------
        // X is a GhB handle object
        //----------------------------------------------------------------------

        matrix->G = gbmx_get_ghb_matrix (X) ;
        CHECK_ERROR (matrix->G == NULL, "invalid GhB matrix") ;
        matrix->will_wait = GB_will_wait (matrix->G) ;
        matrix->nvals = GB_nnz (matrix->G) ; // valid if no pending work
        OK (GrB_Matrix_nrows (&matrix->nrows, matrix->G)) ;
        OK (GrB_Matrix_ncols (&matrix->ncols, matrix->G)) ;
        OK (GxB_Matrix_type (&matrix->type, matrix->G)) ;
        OK (GxB_Type_size (&(matrix->typesize), matrix->type)) ;
        matrix->kind = KIND_GHB ;

    }
    else if (is_grb)
    { 

        //----------------------------------------------------------------------
        // X is a GrB value object
        //----------------------------------------------------------------------

        gbmx_get_grb_matrix (matrix, X) ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // X is a MATLAB matrix
        //----------------------------------------------------------------------

        // get the type and dimensions
        matrix->type = gbmx_mxarray_type (X) ;
        OK (GxB_Type_size (&(matrix->typesize), matrix->type)) ;
        matrix->nrows = (uint64_t) mxGetM (X) ;
        matrix->ncols = (uint64_t) mxGetN (X) ;
        matrix->by_col = true ;
        matrix->nvec_nonempty = -1 ;
        matrix->kind = KIND_BUILTIN ;

        if (matrix->nrows == 0 && matrix->ncols == 0)
        { 

            //------------------------------------------------------------------
            // X is an empty 0-by-0 MATLAB matrix.  X->[pix] are NULL.
            //------------------------------------------------------------------

            matrix->is_empty = true ;
            matrix->sparsity = GxB_FULL ;

        }
        else
        {

            //------------------------------------------------------------------
            // X is a non-empty MATLAB matrix
            //------------------------------------------------------------------

            matrix->sparsity = mxIsSparse (X) ? GxB_SPARSE : GxB_FULL ;
            if (matrix->sparsity == GxB_SPARSE)
            { 
                // X is a sparse MATLAB matrix
                matrix->p = (void *) mxGetJc (X) ;
                matrix->i = (void *) mxGetIr (X) ;
                uint64_t *Xp = (uint64_t *) matrix->p ;
                matrix->nvals = Xp [matrix->ncols] ;
                matrix->plen = matrix->ncols ;
                matrix->nvec = matrix->ncols ;
            }
            else
            { 
                // X is a full MATLAB matrix
                matrix->nvals = matrix->nrows * matrix->ncols ;
            }
            // get the matrix values
            matrix->x = (void *) mxGetData (X) ;
        }
    }
}

