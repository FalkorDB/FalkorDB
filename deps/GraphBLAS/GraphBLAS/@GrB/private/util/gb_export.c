//------------------------------------------------------------------------------
// gb_export: export a GrB_Matrix as a MATLAB matrix or GraphBLAS struct
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// mxArray pargout [0] = gb_export (&C, kind) ; exports C as a MATLAB matrix
// and frees the remaining content of C.

// This function accesses GB_methods inside GraphBLAS.

#include "gb_interface.h"

mxArray *gb_export              // return the exported MATLAB matrix or struct
(
    GrB_Matrix *C_handle,       // GrB_Matrix to export and free
    kind_enum_t kind            // GrB, sparse, full, or built-in
)
{

    //--------------------------------------------------------------------------
    // determine if all entries in C are present
    //--------------------------------------------------------------------------

    uint64_t nrows, ncols ;
    bool is_full = false ;
    if (kind == KIND_BUILTIN || kind == KIND_FULL)
    { 
        uint64_t nvals ;
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
        // export C as a MATLAB sparse matrix
        //----------------------------------------------------------------------

        // Typecast to double, if C is integer (int8, ..., uint64)
        return (gb_export_to_mxsparse (C_handle)) ;

    }
    else if (kind == KIND_FULL)
    { 

        //----------------------------------------------------------------------
        // export C as a MATLAB full matrix, adding explicit zeros if needed
        //----------------------------------------------------------------------

        // No typecasting is needed since MATLAB full matrices support all
        // the same types.

        // ensure C is full
        GrB_Matrix C = NULL ;
        if (!is_full)
        {
            // expand C with explicit zeros so all entries are present
            C = gb_expand_to_full (*C_handle, NULL, GxB_BY_COL, NULL) ;
            OK (GrB_Matrix_free (C_handle)) ;
            (*C_handle) = C ;
            CHECK_ERROR (gb_is_readonly (*C_handle), "internal error 707")
        }

        // ensure the matrix is not readonly
        if (gb_is_readonly (*C_handle))
        {
            // C is shallow so make a deep copy
            OK (GrB_Matrix_dup (&C, *C_handle)) ;
            OK (GrB_Matrix_free (C_handle)) ;
            (*C_handle) = C ;
        }
        CHECK_ERROR (gb_is_readonly (*C_handle), "internal error 717")

        // ensure C is in full format, held by column
        C = (*C_handle) ;
        OK (GrB_Matrix_set_INT32 (C, GxB_FULL,   GxB_SPARSITY_CONTROL)) ;
        OK (GrB_Matrix_set_INT32 (C, GxB_BY_COL, GxB_FORMAT)) ;

        // ensure the matrix is not iso-valued
        OK (GrB_Matrix_set_INT32 (C, 0, GxB_ISO)) ;

        // unload C into the Container and free C
        GxB_Container Container = GB_helper_container ( ) ;
        CHECK_ERROR (Container == NULL, "internal error 911") ;
        OK (GxB_unload_Matrix_into_Container (C, Container, NULL)) ;
        OK (GrB_Matrix_free (C_handle)) ;

        // ensure the container holds the right content: not iso, full, and in
        // column major format.  This is just a sanity check; it should always
        // succeed.
        CHECK_ERROR (Container->iso, "internal error 718") ;
        CHECK_ERROR (Container->format != GxB_FULL, "internal error 719") ;
        CHECK_ERROR (Container->orientation != GrB_COLMAJOR,
            "internal error 720") ;

        // unload the Container->x vector into the raw C array Cx
        void *Cx = NULL ;
        GrB_Type ctype = NULL ;
        uint64_t Cx_size, xlen ;
        int ignore = 0 ;
        OK (GxB_Vector_unload (Container->x, &Cx, &ctype, &xlen, &Cx_size,
            &ignore, NULL)) ;

        // export Cx as a dense nrows-by-ncols MATLAB matrix
        return (gb_export_to_mxfull (&Cx, nrows, ncols, ctype)) ;

    }
    else // kind == KIND_GRB
    { 

        //----------------------------------------------------------------------
        // export C as a MATLAB struct containing a verbatim GrB_Matrix
        //----------------------------------------------------------------------

        // No typecasting is needed since the MATLAB struct can hold all of
        // the opaque content of the GrB_Matrix.
        return (gb_export_to_mxstruct (C_handle)) ;
    }
}

