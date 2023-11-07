//------------------------------------------------------------------------------
// GB_mx_create_full: create a full built-in matrix of a given GrB_Type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

mxArray *GB_mx_create_full      // return new built-in full matrix
(
    const GrB_Index nrows,
    const GrB_Index ncols,
    GrB_Type type               // type of the matrix to create
)
{

    //--------------------------------------------------------------------------
    // allocate an dense matrix of the right type
    //--------------------------------------------------------------------------

    if (type == GrB_BOOL)
    { 
        return (mxCreateLogicalMatrix (nrows, ncols)) ;
    }
    else if (type == GrB_FP32)
    { 
        return (mxCreateNumericMatrix (nrows, ncols, mxSINGLE_CLASS, mxREAL)) ;
    }
    else if (type == GrB_FP64)
    { 
        return (mxCreateNumericMatrix (nrows, ncols, mxDOUBLE_CLASS, mxREAL)) ;
    }
    else if (type == GrB_INT8)
    { 
        return (mxCreateNumericMatrix (nrows, ncols, mxINT8_CLASS, mxREAL)) ;
    }
    else if (type == GrB_INT16)
    { 
        return (mxCreateNumericMatrix (nrows, ncols, mxINT16_CLASS, mxREAL)) ;
    }
    else if (type == GrB_INT32)
    { 
        return (mxCreateNumericMatrix (nrows, ncols, mxINT32_CLASS, mxREAL)) ;
    }
    else if (type == GrB_INT64)
    { 
        return (mxCreateNumericMatrix (nrows, ncols, mxINT64_CLASS, mxREAL)) ;
    }
    else if (type == GrB_UINT8)
    { 
        return (mxCreateNumericMatrix (nrows, ncols, mxUINT8_CLASS, mxREAL)) ;
    }
    else if (type == GrB_UINT16)
    { 
        return (mxCreateNumericMatrix (nrows, ncols, mxUINT16_CLASS, mxREAL)) ;
    }
    else if (type == GrB_UINT32)
    { 
        return (mxCreateNumericMatrix (nrows, ncols, mxUINT32_CLASS, mxREAL)) ;
    }
    else if (type == GrB_UINT64)
    { 
        return (mxCreateNumericMatrix (nrows, ncols, mxUINT64_CLASS, mxREAL)) ;
    }
    else if (type == GxB_FC32)
    {
        return (mxCreateNumericMatrix (nrows, ncols, mxSINGLE_CLASS,
            mxCOMPLEX)) ;
    }
    else if (type == GxB_FC64 || type == Complex)
    {
        return (mxCreateNumericMatrix (nrows, ncols, mxDOUBLE_CLASS,
            mxCOMPLEX)) ;
    }
    else
    {
        mexErrMsgTxt ("unsupported type") ;
        return (NULL) ;
    }
}

