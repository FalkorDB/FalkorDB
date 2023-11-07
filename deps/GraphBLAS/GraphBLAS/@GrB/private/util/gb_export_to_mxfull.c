//------------------------------------------------------------------------------
// gb_export_to_mxfull: export a full array to a built-in full matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input (void *) X is exported to a built-in full mxArray S.

// The input array must be deep, but this cannot be checked here.  The caller
// must ensure that the input X is deep.  The output is a standard built-in full
// matrix as an mxArray.  No typecasting is done.

// mxSetData is used instead of the MATLAB-recommended mxSetDoubles, etc,
// because mxSetData works best for Octave, and it works fine for MATLAB
// since GraphBLAS requires R2018a with the interleaved complex data type.

#include "gb_interface.h"

mxArray *gb_export_to_mxfull    // return exported built-in full matrix F
(
    void **X_handle,            // pointer to array to export
    const GrB_Index nrows,      // dimensions of F
    const GrB_Index ncols,
    GrB_Type type               // type of the array
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    CHECK_ERROR (X_handle == NULL, "internal error 3") ;
    CHECK_ERROR (type == NULL, "internal error 11") ;

    //--------------------------------------------------------------------------
    // allocate an empty full matrix of the right type, then set content
    //--------------------------------------------------------------------------

    mxArray *F ;
    void *X = (*X_handle) ;
    if (X == NULL)
    {
        // A GrB_Matrix C has a null C->x array, if C has no entries.  Since
        // C has already been expanded to a full matrix, C->x can be NULL
        // only if nrows or ncols is zero.
        CHECK_ERROR (nrows > 0 && ncols > 0, "internal error 12") ;
        X = mxMalloc (2 * sizeof (double)) ;
    }

    if (type == GrB_BOOL)
    { 
        F = mxCreateLogicalMatrix (0, 0) ;
    }
    else if (type == GrB_FP32)
    { 
        F = mxCreateNumericMatrix (0, 0, mxSINGLE_CLASS, mxREAL) ;
    }
    else if (type == GrB_FP64)
    { 
        F = mxCreateNumericMatrix (0, 0, mxDOUBLE_CLASS, mxREAL) ;
    }
    else if (type == GrB_INT8)
    { 
        F = mxCreateNumericMatrix (0, 0, mxINT8_CLASS, mxREAL) ;
    }
    else if (type == GrB_INT16)
    { 
        F = mxCreateNumericMatrix (0, 0, mxINT16_CLASS, mxREAL) ;
    }
    else if (type == GrB_INT32)
    { 
        F = mxCreateNumericMatrix (0, 0, mxINT32_CLASS, mxREAL) ;
    }
    else if (type == GrB_INT64)
    { 
        F = mxCreateNumericMatrix (0, 0, mxINT64_CLASS, mxREAL) ;
    }
    else if (type == GrB_UINT8)
    { 
        F = mxCreateNumericMatrix (0, 0, mxUINT8_CLASS, mxREAL) ;
    }
    else if (type == GrB_UINT16)
    { 
        F = mxCreateNumericMatrix (0, 0, mxUINT16_CLASS, mxREAL) ;
    }
    else if (type == GrB_UINT32)
    { 
        F = mxCreateNumericMatrix (0, 0, mxUINT32_CLASS, mxREAL) ;
    }
    else if (type == GrB_UINT64)
    { 
        F = mxCreateNumericMatrix (0, 0, mxUINT64_CLASS, mxREAL) ;
    }
    else if (type == GxB_FC32)
    {
        F = mxCreateNumericMatrix (0, 0, mxSINGLE_CLASS, mxCOMPLEX) ;
    }
    else if (type == GxB_FC64)
    {
        F = mxCreateNumericMatrix (0, 0, mxDOUBLE_CLASS, mxCOMPLEX) ;
    }
    else
    {
        ERROR ("unsupported type") ;
    }

    // set the data
    mxSetData (F, X) ;

    // set the size
    mxSetM (F, nrows) ;
    mxSetN (F, ncols) ;

    // tell the caller that X is exported
    (*X_handle) = NULL ;

    //--------------------------------------------------------------------------
    // return the new built-in full matrix
    //--------------------------------------------------------------------------

    return (F) ;
}

