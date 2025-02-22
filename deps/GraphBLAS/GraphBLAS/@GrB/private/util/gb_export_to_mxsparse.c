//------------------------------------------------------------------------------
// gb_export_to_mxsparse: export a GrB_Matrix to a MATLAB sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input GrB_Matrix A is exported to a MATLAB sparse mxArray S, and freed.

// The input GrB_Matrix A may be shallow or deep.  The output is a standard
// MATLAB sparse matrix as an mxArray.

// This function accesses GB_methods inside GraphBLAS.

#include "gb_interface.h"

mxArray *gb_export_to_mxsparse  // return exported MATLAB sparse matrix S
(
    GrB_Matrix *A_handle        // matrix to export; freed on output
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    CHECK_ERROR (A_handle == NULL || (*A_handle) == NULL, "internal error 2") ;

    //--------------------------------------------------------------------------
    // typecast to a native MATLAB sparse type and free A
    //--------------------------------------------------------------------------

    GrB_Matrix T ;              // T will always be deep
    GrB_Type type ;
    OK (GxB_Matrix_type (&type, *A_handle)) ;
    int fmt ;
    OK (GrB_Matrix_get_INT32 (*A_handle, &fmt, GxB_FORMAT)) ;

    if (fmt == GxB_BY_COL &&
        (type == GrB_BOOL || type == GrB_FP64 || type == GxB_FC64))
    {

        //----------------------------------------------------------------------
        // A is already in a native built-in sparse matrix type, by column
        //----------------------------------------------------------------------

        if (gb_is_readonly (*A_handle))
        { 
            // A is shallow so make a deep copy
            OK (GrB_Matrix_dup (&T, *A_handle)) ;
            OK (GrB_Matrix_free (A_handle)) ;
        }
        else
        { 
            // A is already deep; just transplant it into T
            T = (*A_handle) ;
            (*A_handle) = NULL ;
        }

    }
    else
    { 

        //----------------------------------------------------------------------
        // typecast A to logical, double or double complex, and format by column
        //----------------------------------------------------------------------

        // Built-in sparse matrices can only be logical, double, or double
        // complex.  These correspond to GrB_BOOL, GrB_FP64, and GxB_FC64,
        // respectively.  A is typecasted to logical, double or double complex,
        // and converted to CSC format if not already in that format.

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

        T = gb_typecast (*A_handle, type, GxB_BY_COL, GxB_SPARSE) ;

        OK (GrB_Matrix_free (A_handle)) ;
    }

    // ensure T is deep
    CHECK_ERROR (gb_is_readonly (T), "internal error 7") ;

    //--------------------------------------------------------------------------
    // drop zeros from T
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
    GrB_Scalar zero ;
    OK (GrB_Scalar_new (&zero, type)) ;
    OK (GrB_Scalar_setElement_FP64 (zero, 0)) ;
    OK1 (T, GrB_Matrix_select_Scalar (T, NULL, NULL, op, T, zero, NULL)) ;
    OK (GrB_Scalar_free (&zero)) ;

    //--------------------------------------------------------------------------
    // create the new built-in sparse matrix
    //--------------------------------------------------------------------------

    uint64_t nrows, ncols, nvals ;
    OK (GrB_Matrix_nvals (&nvals, T)) ;
    OK (GrB_Matrix_nrows (&nrows, T)) ;
    OK (GrB_Matrix_ncols (&ncols, T)) ;

    mxArray *S ;

    if (nvals == 0)
    {

        //----------------------------------------------------------------------
        // allocate an empty sparse matrix of the right type and size
        //----------------------------------------------------------------------

        if (type == GrB_BOOL)
        { 
            S = mxCreateSparseLogicalMatrix (nrows, ncols, 1) ;
        }
        else if (type == GxB_FC64)
        { 
            S = mxCreateSparse (nrows, ncols, 1, mxCOMPLEX) ;
        }
        else
        { 
            S = mxCreateSparse (nrows, ncols, 1, mxREAL) ;
        }
        OK (GrB_Matrix_free (&T)) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // export the content of T as a sparse CSC matrix (all-64-bit)
        //----------------------------------------------------------------------

        uint64_t Tp_size, Ti_size, Tx_size, type_size, plen, ilen, xlen ;
        uint64_t *Tp, *Ti ;
        void *Tx ;

        // ensure the matrix is in sparse CSC format
        OK (GrB_Matrix_set_INT32 (T, GxB_SPARSE, GxB_SPARSITY_CONTROL)) ;
        OK (GrB_Matrix_set_INT32 (T, GxB_BY_COL, GxB_FORMAT)) ;

        // ensure the matrix uses all 64-bit integers
        OK (GrB_Matrix_set_INT32 (T, 64, GxB_ROWINDEX_INTEGER_HINT)) ;
        OK (GrB_Matrix_set_INT32 (T, 64, GxB_COLINDEX_INTEGER_HINT)) ;
        OK (GrB_Matrix_set_INT32 (T, 64, GxB_OFFSET_INTEGER_HINT)) ;

        // ensure the matrix is not iso-valued
        OK (GrB_Matrix_set_INT32 (T, 0, GxB_ISO)) ;

        // unload T into a Container and free T
        GxB_Container Container = GB_helper_container ( ) ;
        CHECK_ERROR (Container == NULL, "internal error 911") ;
        OK (GxB_unload_Matrix_into_Container (T, Container, NULL)) ;
        OK (GrB_Matrix_free (&T)) ;

        // ensure the container holds content that is not jumbled or iso,
        // and is in sparse CSC format; this 'cannot' fail but check just
        // in case.
        CHECK_ERROR (Container->iso, "internal error 904") ;
        CHECK_ERROR (Container->jumbled, "internal error 905") ;
        CHECK_ERROR (Container->format != GxB_SPARSE, "internal error 906") ;
        CHECK_ERROR (Container->orientation != GrB_COLMAJOR,
            "internal error 907") ;

        // unload the Container GrB_Vectors into raw C arrays Tp, Ti, and Tx
        GrB_Type Tp_type, Ti_type, Tx_type ;
        int ignore = 0 ;
        OK (GxB_Vector_unload (Container->p, (void **) &Tp, &Tp_type, &plen,
            &Tp_size, &ignore, NULL)) ;
        OK (GxB_Vector_unload (Container->i, (void **) &Ti, &Ti_type, &ilen,
            &Ti_size, &ignore, NULL)) ;
        OK (GxB_Vector_unload (Container->x, (void **) &Tx, &Tx_type, &xlen,
            &Tx_size, &ignore, NULL)) ;

        // ensure the types are correct; this 'cannot' fail but check anyway
        CHECK_ERROR (Tp_type != GrB_UINT64, "internal error 901") ;
        CHECK_ERROR (Ti_type != GrB_UINT64, "internal error 902") ;
        CHECK_ERROR (Tx_type != type, "internal error 903") ;

        //----------------------------------------------------------------------
        // allocate an empty sparse matrix of the right type, then set content
        //----------------------------------------------------------------------

        if (type == GrB_BOOL)
        { 
            S = mxCreateSparseLogicalMatrix (0, 0, 1) ;
            type_size = 1 ;
        }
        else if (type == GxB_FC64)
        { 
            S = mxCreateSparse (0, 0, 1, mxCOMPLEX) ;
            type_size = 16 ;
        }
        else // type == GrB_FP64
        { 
            S = mxCreateSparse (0, 0, 1, mxREAL) ;
            type_size = 8 ;
        }

        // set the size
        mxSetM (S, nrows) ;
        mxSetN (S, ncols) ;
        int64_t nzmax = MIN (Ti_size / sizeof (int64_t), Tx_size / type_size) ;
        mxSetNzmax (S, nzmax) ;

        // set the column pointers
        void *p = mxGetJc (S) ; gb_mxfree (&p) ;
        mxSetJc (S, (mwIndex *) Tp) ;

        // set the row indices
        p = mxGetIr (S) ; gb_mxfree (&p) ;
        mxSetIr (S, (mwIndex *) Ti) ;

        // set the values
        // use mxGetData and mxSetData (best for Octave, fine for MATLAB)
        p = mxGetData (S) ; gb_mxfree (&p) ;
        mxSetData (S, Tx) ;
    }

    //--------------------------------------------------------------------------
    // return the new built-in MATLAB sparse matrix
    //--------------------------------------------------------------------------

    return (S) ;
}

