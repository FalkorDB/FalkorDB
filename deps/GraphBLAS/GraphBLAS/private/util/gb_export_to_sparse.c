//------------------------------------------------------------------------------
// gb_export_to_sparse: prepare a GrB_Matrix to become a MATLAB sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input GrB_Matrix C is being exported to a G.opaque handle, to become a
// GrB object.  This method modifies its format and integer sizes to be
// directly compatible with a MATLAB sparse matrix.  After the caller
// mexFunction finishes, another mexFunction (gbmex_builtin) will copy G into a
// proper MATLAB sparse matrix.

// No mx* methods are called, so that any memory allocation failures can be
// properly handled.

#undef  FREE_WORK
#define FREE_WORK                   \
    GrB_Scalar_free (&zero) ;       \
    GrB_Matrix_free (&T) ;

#undef  FREE_ALL
#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (C_handle) ;

GrB_Info gb_export_to_sparse
(
    // input/output
    GrB_Matrix *C_handle,   // GraphBLAS matrix to modify for export to MATLAB
    // intput
    const int arena,
    char err [ERRLEN]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, T = NULL ;
    GrB_Scalar zero = NULL ;
    CHECK_ERROR (C_handle == NULL || (*C_handle) == NULL, "internal error 16") ;

    //--------------------------------------------------------------------------
    // typecast to a native MATLAB sparse type
    //--------------------------------------------------------------------------

    C = (*C_handle) ;
    GrB_Type type ;
    OK (GxB_Matrix_type (&type, C)) ;
    int fmt ;
    OK (GrB_Matrix_get_INT32 (C, &fmt, GxB_FORMAT)) ;

    if (! (fmt == GxB_BY_COL &&
        (type == GrB_BOOL || type == GrB_FP64 || type == GxB_FC64)))
    {

        //----------------------------------------------------------------------
        // typecast C to logical, double or double complex, and format by column
        //----------------------------------------------------------------------

        // Built-in MATLAB sparse matrices can only be logical, double, or
        // double complex.  These correspond to GrB_BOOL, GrB_FP64, and
        // GxB_FC64, respectively.  C is typecasted to logical, double or
        // double complex, and converted to CSC format if not already in that
        // format.

        // FUTURE: recent versions of MATLAB (R2025a and later) support
        // GrB_FP32 and GxB_FC32 sparse matrices.  Check for the version of
        // MATLAB and exploit those data types.

        if (type == GxB_FC32 || type == GxB_FC64)
        { 
            // typecast to double complex, by col
            type = GxB_FC64 ;
        }
        else if (type == GrB_BOOL)
        { 
            // typecast to logical, by col
            type = GrB_BOOL ;
        }
        else
        { 
            // typecast to double, by col
            type = GrB_FP64 ;
        }

        OK (gb_typecast (&T, C, type, GxB_BY_COL, GxB_SPARSE, arena, err)) ;
        GrB_Matrix_free (C_handle) ;
        (*C_handle) = T ;
        T = NULL ;
        C = (*C_handle) ;
    }

    //--------------------------------------------------------------------------
    // drop zeros from C
    //--------------------------------------------------------------------------

    GrB_IndexUnaryOp op ;
    if (type == GrB_BOOL)
    { 
        op = GrB_VALUENE_BOOL ;
    }
    else if (type == GrB_FP64)
    { 
        op = GrB_VALUENE_FP64 ;
    }
    else if (type == GxB_FC64)
    { 
        op = GxB_VALUENE_FC64 ;
    }
    OK (GxB_Scalar_new_arena (&zero, type, arena, arena)) ;
    OK (GrB_Scalar_setElement_FP64 (zero, 0)) ;
    OK1 (C, GrB_Matrix_select_Scalar (C, NULL, NULL, op, C, zero, NULL)) ;

    //--------------------------------------------------------------------------
    // ensure C has the correct properties
    //--------------------------------------------------------------------------

    // ensure the matrix is in sparse CSC format
    OK (GrB_Matrix_set_INT32 (C, GxB_SPARSE, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_Matrix_set_INT32 (C, GxB_BY_COL, GxB_FORMAT)) ;

    // ensure the matrix uses all 64-bit integers
    OK (GrB_Matrix_set_INT32 (C, 64, GxB_ROWINDEX_INTEGER_HINT)) ;
    OK (GrB_Matrix_set_INT32 (C, 64, GxB_COLINDEX_INTEGER_HINT)) ;
    OK (GrB_Matrix_set_INT32 (C, 64, GxB_OFFSET_INTEGER_HINT)) ;

    // ensure the matrix is not iso-valued
    OK (GrB_Matrix_set_INT32 (C, 0, GxB_ISO)) ;

    //--------------------------------------------------------------------------
    // finalize the matrix
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_wait (C, GrB_MATERIALIZE)) ;

    //--------------------------------------------------------------------------
    // free workspace and return results
    //--------------------------------------------------------------------------

    FREE_WORK ;
    return (GrB_SUCCESS) ;
}

#undef  FREE_WORK
#define FREE_WORK
#undef  FREE_ALL
#define FREE_ALL

