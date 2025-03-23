//------------------------------------------------------------------------------
// gb_mxarray_to_list: return GrB_Vector for assign, subassign, extract, build
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

//------------------------------------------------------------------------------
// gb_subtract_base:  V = S, or V = S-1
//------------------------------------------------------------------------------

static GrB_Vector gb_subtract_base
(
    GrB_Vector *S,              // input, returned as V or freed for V=S-1
    const int base_offset       // 1 or 0
)
{
    GrB_Vector V = NULL ;
    if (base_offset == 0)
    { 
        // V = S, with no change of type
        V = (GrB_Vector) (*S) ;
        (*S) = NULL ;
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
        OK (GrB_Vector_new (&V, type, n)) ;
        ASSERT_VECTOR_OK (V, "V result, before apply", GB0) ;
        ASSERT_VECTOR_OK (*S, "S before apply", GB0) ;
        OK (GrB_Vector_apply_BinaryOp2nd_UINT64 (V, NULL, NULL, minus, *S, 1,
            NULL)) ;
        ASSERT_VECTOR_OK (V, "V result, after apply", GB0) ;
        GrB_Vector_free (S) ;
    }
    return (V) ;
}

//------------------------------------------------------------------------------
// gb_mxarray_to_list
//------------------------------------------------------------------------------

GrB_Vector gb_mxarray_to_list      // list of indices or values
(
    const mxArray *X,       // MATLAB input matrix or struct with GrB content
    const int base_offset   // 1 or 0
)
{ 

    //--------------------------------------------------------------------------
    // get a shallow GrB_Matrix S of the input MATLAB matrix or struct
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL ;
    GrB_Vector V = NULL ;
    GrB_Matrix S = gb_get_shallow (X) ;

    //--------------------------------------------------------------------------
    // get the properties of S
    //--------------------------------------------------------------------------

    uint64_t ncols, nrows ;
    OK (GrB_Matrix_nrows (&nrows, S)) ;
    OK (GrB_Matrix_ncols (&ncols, S)) ;

    //--------------------------------------------------------------------------
    // check for quick return
    //--------------------------------------------------------------------------

    if (ncols == 0 || nrows == 0)
    { 
        // return a zero-length vector
        GrB_Type type ;
        OK (GxB_Matrix_type (&type, S)) ;
        OK (GrB_Vector_new (&V, type, 0)) ;
        ASSERT_VECTOR_OK (V, "V result, empty", GB0) ;
        GrB_Matrix_free (&S) ;
        return (V) ;
    }

    int sparsity, fmt ;
    OK (GrB_Matrix_get_INT32 (S, &fmt, GxB_FORMAT)) ;
    OK (GrB_Matrix_get_INT32 (S, &sparsity, GxB_SPARSITY_STATUS)) ;
    bool quick = false ;

    if (ncols == 1 && sparsity != GxB_HYPERSPARSE && fmt == GxB_BY_COL)
    { 
        // return S as a shallow GrB_Vector
        quick = true ;
    }

    if (nrows == 1 && sparsity != GxB_HYPERSPARSE && fmt == GxB_BY_ROW)
    { 
        // quick in-place transpose, by converting it to by-column
        quick = true ;
        S->is_csc = true ;
    }

    if (quick)
    { 
        // return S as a shallow GrB_Vector, but subtract the base if needed
        ASSERT (gb_is_column_vector (S)) ;
        ASSERT_VECTOR_OK ((GrB_Vector) S, "S as vector", GB0) ;
        // V = S - base_offset
        V = gb_subtract_base ((GrB_Vector *) &S, base_offset) ;
        ASSERT_VECTOR_OK (V, "V result, quick", GB0) ;
        return (V) ;
    }

    //--------------------------------------------------------------------------
    // reshape S into (nrows*ncols)-by-1 and return it as a GrB_Vector
    //--------------------------------------------------------------------------

    // C = S (:)
    if (((double) nrows) * ((double) ncols) > INT64_MAX / 8)
    { 
        ERROR ("input matrix dimensions are too large") ;
    }
    OK (GxB_Matrix_reshapeDup (&C, S, true, nrows * ncols, 1, NULL)) ;
    GrB_Matrix_free (&S) ;

    // ensure C is not hypersparse, and is stored by column
    OK (GrB_Matrix_set_INT32 (C, GxB_SPARSE + GxB_BITMAP + GxB_FULL,
        GxB_SPARSITY_CONTROL)) ;
    OK (GrB_Matrix_set_INT32 (C, GxB_BY_COL, GxB_FORMAT)) ;

    // C is now a valid column vector
    ASSERT (gb_is_column_vector (C)) ;

    // V = C - base_offset
    V = gb_subtract_base ((GrB_Vector *) &C, base_offset) ;

    // V is now a valid GrB_Vector (no longer shallow)
    ASSERT_VECTOR_OK (V, "V result, slow", GB0) ;
    return (V) ;
}

