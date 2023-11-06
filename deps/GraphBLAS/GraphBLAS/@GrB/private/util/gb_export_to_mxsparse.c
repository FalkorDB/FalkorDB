//------------------------------------------------------------------------------
// gb_export_to_mxsparse: export a GrB_Matrix to a built-in sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input GrB_Matrix A is exported to a built-in sparse mxArray S, and freed.

// The input GrB_Matrix A may be shallow or deep.  The output is a standard
// built-in sparse matrix as an mxArray.

#include "gb_interface.h"

mxArray *gb_export_to_mxsparse  // return exported built-in sparse matrix S
(
    GrB_Matrix *A_handle        // matrix to export; freed on output
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    CHECK_ERROR (A_handle == NULL || (*A_handle) == NULL, "internal error 2") ;

    //--------------------------------------------------------------------------
    // typecast to a native built-in sparse type and free A
    //--------------------------------------------------------------------------

    GrB_Matrix T ;              // T will always be deep
    GrB_Type type ;
    OK (GxB_Matrix_type (&type, *A_handle)) ;
    GxB_Format_Value fmt ;
    OK (GxB_Matrix_Option_get (*A_handle, GxB_FORMAT, &fmt)) ;

    if (fmt == GxB_BY_COL &&
        (type == GrB_BOOL || type == GrB_FP64 || type == GxB_FC64))
    {

        //----------------------------------------------------------------------
        // A is already in a native built-in sparse matrix type, by column
        //----------------------------------------------------------------------

        if (GB_is_shallow (*A_handle))
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
    CHECK_ERROR (GB_is_shallow (T), "internal error 7") ;

    //--------------------------------------------------------------------------
    // drop zeros from T
    //--------------------------------------------------------------------------

    OK1 (T, GxB_Matrix_select (T, NULL, NULL, GxB_NONZERO, T, NULL, NULL)) ;

    //--------------------------------------------------------------------------
    // create the new built-in sparse matrix
    //--------------------------------------------------------------------------

    GrB_Index nrows, ncols, nvals ;
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
        // export the content of T as a sparse CSC matrix
        //----------------------------------------------------------------------

        GrB_Index Tp_size, Ti_size, Tx_size, type_size ;
        uint64_t *Tp, *Ti ;
        void *Tx ;

        // pass jumbled as NULL to indicate the matrix must be sorted
        // pass iso as NULL to indicate it cannot be uniform valued
        OK (GxB_Matrix_export_CSC (&T, &type, &nrows, &ncols,
            &Tp, &Ti, &Tx, &Tp_size, &Ti_size, &Tx_size, NULL, NULL, NULL)) ;

        CHECK_ERROR (Ti_size == 0, "internal error 8") ;
        CHECK_ERROR (Tp == NULL || Ti == NULL || Tx == NULL,
            "internal error 9") ;

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
        int64_t nzmax = GB_IMIN (Ti_size / sizeof (int64_t),
                                 Tx_size / type_size) ;
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
    // return the new built-in sparse matrix
    //--------------------------------------------------------------------------

    return (S) ;
}

