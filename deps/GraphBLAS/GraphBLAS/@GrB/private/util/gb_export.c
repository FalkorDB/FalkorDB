//------------------------------------------------------------------------------
// gb_export: export a GrB_Matrix as a built-in matrix or GraphBLAS struct
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// mxArray pargout [0] = gb_export (&C, kind) ; exports C as a built-in matrix
// and frees the remaining content of C.

#include "gb_interface.h"

mxArray *gb_export              // return the exported built-in matrix or struct
(
    GrB_Matrix *C_handle,       // GrB_Matrix to export and free
    kind_enum_t kind            // GrB, sparse, full, or built-in
)
{

    //--------------------------------------------------------------------------
    // determine if all entries in C are present
    //--------------------------------------------------------------------------

    GrB_Index nrows, ncols ;
    bool is_full = false ;
    if (kind == KIND_BUILTIN || kind == KIND_FULL)
    { 
        GrB_Index nvals ;
        OK (GrB_Matrix_nvals (&nvals, *C_handle)) ;
        OK (GrB_Matrix_nrows (&nrows, *C_handle)) ;
        OK (GrB_Matrix_ncols (&ncols, *C_handle)) ;
        is_full = ((double) nrows * (double) ncols == (double) nvals) ;
    }

    if (kind == KIND_BUILTIN)
    { 
        // export as full if all entries present, or sparse otherwise
        kind = (is_full) ? KIND_FULL : KIND_SPARSE ;
    }

    //--------------------------------------------------------------------------
    // export the matrix
    //--------------------------------------------------------------------------

    if (kind == KIND_SPARSE)
    { 

        //----------------------------------------------------------------------
        // export C as a built-in sparse matrix
        //----------------------------------------------------------------------

        // Typecast to double, if C is integer (int8, ..., uint64)
        return (gb_export_to_mxsparse (C_handle)) ;

    }
    else if (kind == KIND_FULL)
    { 

        //----------------------------------------------------------------------
        // export C as a built-in full matrix, adding explicit zeros if needed
        //----------------------------------------------------------------------

        // No typecasting is needed since built-in full matrices support all
        // the same types.

        GrB_Matrix C = NULL ;
        if (!is_full)
        {
            // expand C with explicit zeros so all entries are present
            C = gb_expand_to_full (*C_handle, NULL, GxB_BY_COL, NULL) ;
            OK (GrB_Matrix_free (C_handle)) ;
            (*C_handle) = C ;
            CHECK_ERROR (GB_is_shallow (*C_handle), "internal error 707")
        }

        if (GB_is_shallow (*C_handle))
        {
            // C is shallow so make a deep copy
            OK (GrB_Matrix_dup (&C, *C_handle)) ;
            OK (GrB_Matrix_free (C_handle)) ;
            (*C_handle) = C ;
        }

        CHECK_ERROR (GB_is_shallow (*C_handle), "internal error 717")

        // export as a full matrix, held by column, not uniform-valued
        void *Cx = NULL ;
        GrB_Type ctype = NULL ;
        GrB_Index Cx_size ;
        OK (GxB_Matrix_export_FullC (C_handle, &ctype, &nrows, &ncols,
            &Cx, &Cx_size, NULL, NULL)) ;

        return (gb_export_to_mxfull (&Cx, nrows, ncols, ctype)) ;

    }
    else // kind == KIND_GRB
    { 

        //----------------------------------------------------------------------
        // export C as a built-in struct containing a verbatim GrB_Matrix
        //----------------------------------------------------------------------

        // No typecasting is needed since the built-in struct can hold all of
        // the opaque content of the GrB_Matrix.
        return (gb_export_to_mxstruct (C_handle)) ;
    }
}

