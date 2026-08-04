//------------------------------------------------------------------------------
// gb_matrix_to_list: return GrB_Vector for assign, subassign, extract, build
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// gb_subtract_base:  V = S, or V = S-1
//------------------------------------------------------------------------------

#undef  FREE_ALL
#define FREE_ALL                            \
    GrB_Vector_free (V_to_free) ;

static GrB_Info gb_subtract_base
(
    // output:
    GrB_Vector *V,              // must not be freed by the caller
    GrB_Vector *V_to_free,      // NULL, or S_to_free on output;
                                // must be freed by the caller
    // input/output:
    GrB_Vector *S,              // set to NULL on output
    GrB_Vector *S_to_free,      // set to NULL on output
    // input:
    const int base_offset,      // 1 or 0
    const int arena,
    char err [ERRLEN]
)
{
    (*V) = NULL ;
    (*V_to_free) = NULL ;
    if (base_offset == 0)
    { 
        // V = S, with no change of type
        (*V) = (*S) ;
        (*S) = NULL ;
        (*V_to_free) = (*S_to_free) ;
        (*S_to_free) = NULL ;
    }
    else
    { 
        // V = S-1, but typecast to uint32 or uint64 to avoid roundoff errors
        GrB_Type type ;
        OK (GxB_Vector_type (&type, *S)) ;
        GrB_BinaryOp minus ;
        if (type == GrB_BOOL   || type == GrB_INT8  || type == GrB_INT16  ||
            type == GrB_INT32  || type == GrB_UINT8 || type == GrB_UINT16 ||
            type == GrB_UINT32 || type == GrB_FP32  || type == GxB_FC32)
        { 
            type = GrB_UINT32 ;
            minus = GrB_MINUS_UINT32 ;
        }
        else
        { 
            type = GrB_UINT64 ;
            minus = GrB_MINUS_UINT64 ;
        }
        uint64_t n ;
        OK (GrB_Vector_size (&n, *S)) ;
        OK (GxB_Vector_new_arena (V_to_free, type, n, arena, arena)) ;
        ASSERT_VECTOR_OK (*S, "S before apply", GB0) ;
        OK (GrB_Vector_apply_BinaryOp2nd_UINT64 (*V_to_free, NULL, NULL, minus,
            *S, 1, NULL)) ;
        ASSERT_VECTOR_OK (*V_to_free, "V result, after apply", GB0) ;
        (*V) = (*V_to_free) ;
        (*S) = NULL ;
        GrB_Vector_free (S_to_free) ;
    }

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// gb_matrix_to_list
//------------------------------------------------------------------------------

#undef  FREE_WORK
#define FREE_WORK                           \
    GrB_Vector_free (&C_to_free) ;          \
    GrB_Matrix_free (&S_to_free) ;

#undef  FREE_ALL
#define FREE_ALL                            \
    FREE_WORK ;                             \
    GrB_Matrix_free (&C) ;                  \
    GrB_Vector_free (&V_to_free) ;

GrB_Info gb_matrix_to_list
(
    // outputs:
    GrB_Vector *V_handle,   // list of indices or values; caller must not free
    GrB_Vector *V_to_free_handle,  // must be freed by the caller
    // inputs:
    gb_matrix matrix,
    const int base_offset,  // 1 or 0
    const int arena,
    char err [ERRLEN]
)
{ 

    //--------------------------------------------------------------------------
    // get a shallow GrB_Matrix S of the input MATLAB matrix or struct
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, S = NULL, S_to_free = NULL ;
    GrB_Vector V = NULL, V_to_free = NULL, C_to_free = NULL ;
    (*V_handle) = NULL ;
    (*V_to_free_handle) = NULL ;

    OK (gb_get_matrix (&S, &S_to_free, matrix, arena, err)) ;

    //--------------------------------------------------------------------------
    // get the properties of S
    //--------------------------------------------------------------------------

    uint64_t ncols, nrows ;
    int sparsity, fmt ;
    bool is_column_vector ;

    OK (GrB_Matrix_nrows (&nrows, S)) ;
    OK (GrB_Matrix_ncols (&ncols, S)) ;
    OK (GrB_Matrix_get_INT32 (S, &fmt, GxB_FORMAT)) ;
    OK (GrB_Matrix_get_INT32 (S, &sparsity, GxB_SPARSITY_STATUS)) ;

    //--------------------------------------------------------------------------
    // construct the vector V containing the list
    //--------------------------------------------------------------------------

    if (ncols == 0 || nrows == 0)
    { 

        //----------------------------------------------------------------------
        // return a zero-length vector
        //----------------------------------------------------------------------

        GrB_Type type ;
        OK (GxB_Matrix_type (&type, S)) ;
        OK (GxB_Vector_new_arena (&V, type, 0, arena, arena)) ;
        V_to_free = V ;
        ASSERT_VECTOR_OK (V, "V result, empty", GB0) ;

    }
    else if (ncols == 1 && sparsity != GxB_HYPERSPARSE && fmt == GxB_BY_COL)
    { 

        //----------------------------------------------------------------------
        // return S as a shallow GrB_Vector, but subtract the base if needed
        //----------------------------------------------------------------------

        OK (gb_is_column_vector (&is_column_vector, S, err)) ;
        ASSERT (is_column_vector) ;
        ASSERT_VECTOR_OK ((GrB_Vector) S, "S as vector", GB0) ;
        // V = S - base_offset
        OK (gb_subtract_base (&V, &V_to_free,
            (GrB_Vector *) &S, (GrB_Vector *) &S_to_free, base_offset, arena,
            err)) ;
        ASSERT_VECTOR_OK (V, "V result, quick", GB0) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // reshape S into (nrows*ncols)-by-1 and return it as a GrB_Vector
        //----------------------------------------------------------------------

        // C = S (:)
        if (((double) nrows) * ((double) ncols) > (double) (INT64_MAX / 8))
        { 
            ERROR ("input matrix dimensions are too large",
                GrB_DIMENSION_MISMATCH) ;
        }
        OK (GxB_Matrix_reshapeDup_arena (&C, S, true, nrows * ncols, 1,
            arena, arena, NULL)) ;
        GrB_Matrix_free (&S_to_free) ;

        // ensure C is not hypersparse, and is stored by column
        OK (GrB_Matrix_set_INT32 (C, GxB_SPARSE + GxB_BITMAP + GxB_FULL,
            GxB_SPARSITY_CONTROL)) ;
        OK (GrB_Matrix_set_INT32 (C, GxB_BY_COL, GxB_FORMAT)) ;

        // C is now a valid column vector
        OK (gb_is_column_vector (&is_column_vector, C, err)) ;
        ASSERT (is_column_vector) ;

        // V = C - base_offset
        C_to_free = (GrB_Vector) C ;
        OK (gb_subtract_base (&V, &V_to_free,
            (GrB_Vector *) &C, &C_to_free, base_offset, arena, err)) ;

        // V is now a valid GrB_Vector; must be freed by the caller
        ASSERT_VECTOR_OK (V, "V result, slow", GB0) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    (*V_handle) = V ;
    (*V_to_free_handle) = V_to_free ;
    return (GrB_SUCCESS) ;
}

#undef  FREE_WORK
#define FREE_WORK
#undef  FREE_ALL
#define FREE_ALL

