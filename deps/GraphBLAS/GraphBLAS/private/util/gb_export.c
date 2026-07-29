//------------------------------------------------------------------------------
// gb_export: export a GrB_Matrix as a GraphBLAS GrB handle or GrB matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gb_export prepares C for export as a GrB or GhB matrix object for MATLAB,
// with 4 possible kinds:
//
// KIND_GRB or KIND_GHB     C will remain a GrB or GhB matrix object
// KIND_SPARSE     C will become a built-in MATLAB/Octave sparse matrix
// KIND_FULL       C will become a built-in MATLAB/Octave full matrix
// KIND_BUILTIN    C will become a built-in MATLAB/Octave sparse or full matrix
//
// If kind is KIND_GRB or KIND_GHB, the matrix will remain a GrB or GhB matrix
// object.  Otherwise, it is exported as a GrB_Matrix (handle or struct) with
// properties that match a sparse or full MATLAB/Octave matrix.  It is then
// directly copied into a MATLAB/Octave matrix in a subsequent call to the
// gbmex_builtin mexFunction.

#undef  FREE_WORK
#define FREE_WORK                   \
    GrB_Matrix_free (&T) ;

#undef  FREE_ALL
#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (C_handle) ;

GrB_Info gb_export              // export a GrB_Matrix to MATLAB
(
    // output:
    GrB_Matrix *C_opaque,       // matrix for export as GhB
    // input/output:
    GrB_Matrix *C_handle,       // GrB_Matrix to export
    // input:
    kind_enum_t kind,           // GrB, sparse, full, or built-in
    const bool ghb,
    char err [ERRLEN]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, T = NULL ;
    int arena = ghb ? GrB_DEFAULT : MXARENA ;
    CHECK_ERROR (C_handle == NULL || (*C_handle == NULL), "internal error 13") ;
    C = (*C_handle) ;

    //--------------------------------------------------------------------------
    // ensure C has no readonly components
    //--------------------------------------------------------------------------

    int readonly ;
    OK (GrB_Matrix_get_INT32 (C, &readonly, GxB_IS_READONLY)) ;

    if (readonly)
    { 
        // C has readonly components so make a deep copy
        OK (GxB_Matrix_dup_arena (&T, C, arena, arena)) ;
        GrB_Matrix_free (C_handle) ;
        (*C_handle) = T ;
        T = NULL ;
        C = (*C_handle) ;
    }

    //--------------------------------------------------------------------------
    // for GrB value matrices, ensure C has no pending work
    //--------------------------------------------------------------------------

    if (!ghb)
    { 
        OK (GrB_Matrix_wait (C, GrB_MATERIALIZE)) ;
    }

    //--------------------------------------------------------------------------
    // determine if all entries in C are present
    //--------------------------------------------------------------------------

    if (kind == KIND_BUILTIN)
    { 
        // export as full if all entries present, or sparse otherwise
        uint64_t nrows, ncols, nvals ;
        OK (GrB_Matrix_nvals (&nvals, C)) ;
        OK (GrB_Matrix_nrows (&nrows, C)) ;
        OK (GrB_Matrix_ncols (&ncols, C)) ;
        bool is_full = ((double) nrows * (double) ncols == (double) nvals) ;
        kind = (is_full) ? KIND_FULL : KIND_SPARSE ;
    }

    //--------------------------------------------------------------------------
    // conform the matrix to a MATLAB sparse or full format, if requested
    //--------------------------------------------------------------------------

    if (kind == KIND_SPARSE)
    { 

        //----------------------------------------------------------------------
        // C will become a MATLAB sparse matrix
        //----------------------------------------------------------------------

        // Typecast to double, if C is integer (int8, ..., uint64)
        OK (gb_export_to_sparse (C_handle, arena, err)) ;
        C = (*C_handle) ;

    }
    else if (kind == KIND_FULL)
    { 

        //----------------------------------------------------------------------
        // C will become a MATLAB full matrix
        //----------------------------------------------------------------------

        OK (gb_export_to_full (C_handle, arena, err)) ;
        C = (*C_handle) ;
    }

    //--------------------------------------------------------------------------
    // export the result
    //--------------------------------------------------------------------------

    // C should now be deep, but double-check here
    OK (GrB_Matrix_get_INT32 (C, &readonly, GxB_IS_READONLY)) ;
    CHECK_ERROR (readonly, "internal error 14") ;

    if (C_opaque != NULL)
    {
        // export the GhB handle to the output
        (*C_opaque) = C ;
        (*C_handle) = NULL ;
    }
    return (GrB_SUCCESS) ;
}

#undef  FREE_WORK
#define FREE_WORK
#undef  FREE_ALL
#define FREE_ALL

