//------------------------------------------------------------------------------
// gb_export_to_full: prepare a GrB_Matrix to become a MATLAB full matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input GrB Matrix C is being exported to a G.opaque handle, to become a
// GrB object.  This method modifies its format to be directly compatible with
// a MATLAB full matrix.  After the caller mexFunction finishes, another
// mexFunction will copy G into a proper MATLAB full matrix, if desired.

// No mx* methods are called, so that any memory allocation failures can
// be properly handled.

#undef  FREE_WORK
#define FREE_WORK                   \
    GrB_Matrix_free (&T) ;

#undef  FREE_ALL
#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (C_handle) ;

GrB_Info gb_export_to_full
(
    GrB_Matrix *C_handle,   // GraphBLAS matrix to modify for export to MATLAB
    const int arena,
    char err [ERRLEN]
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, T = NULL ;
    CHECK_ERROR (C_handle == NULL || (*C_handle == NULL), "internal error 15") ;
    C = (*C_handle) ;

    //--------------------------------------------------------------------------
    // determine if all entries in C are present
    //--------------------------------------------------------------------------

    uint64_t nrows, ncols, nvals ;
    OK (GrB_Matrix_nvals (&nvals, C)) ;
    OK (GrB_Matrix_nrows (&nrows, C)) ;
    OK (GrB_Matrix_ncols (&ncols, C)) ;
    bool is_full = ((double) nrows * (double) ncols == (double) nvals) ;

    //--------------------------------------------------------------------------
    // expand the matrix to full, if needed
    //--------------------------------------------------------------------------

    // No typecasting is needed since MATLAB full matrices support all
    // the same types.

    // ensure C is full
    if (!is_full)
    { 
        // expand C with explicit zeros so all entries are present
        OK (gb_expand_to_full (&T, C, NULL, GxB_BY_COL, NULL, arena, err)) ;
        GrB_Matrix_free (C_handle) ;
        (*C_handle) = T ;
        T = NULL ;
        C = (*C_handle) ;
    }

    //--------------------------------------------------------------------------
    // ensure C has the right properties
    //--------------------------------------------------------------------------

    // ensure C is in full format, held by column
    OK (GrB_Matrix_set_INT32 (C, GxB_FULL,   GxB_SPARSITY_CONTROL)) ;
    OK (GrB_Matrix_set_INT32 (C, GxB_BY_COL, GxB_FORMAT)) ;

    // ensure the matrix is not iso-valued
    OK (GrB_Matrix_set_INT32 (C, 0, GxB_ISO)) ;

    //--------------------------------------------------------------------------
    // finalize the matrix
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_wait (C, GrB_MATERIALIZE)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    return (GrB_SUCCESS) ;
}

#undef  FREE_WORK
#define FREE_WORK
#undef  FREE_ALL
#define FREE_ALL

