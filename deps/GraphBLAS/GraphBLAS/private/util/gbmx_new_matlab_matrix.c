//------------------------------------------------------------------------------
// gbmx_new_matlab_matrix: create a full MATLAB matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

mxArray *gbmx_new_matlab_matrix // return new MATLAB full matrix
(
    const uint64_t nrows,       // dimensions
    const uint64_t ncols,
    GrB_Type type               // type of the array
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    CHECK_ERROR (type == NULL, "internal error 18") ;

    //--------------------------------------------------------------------------
    // allocate a full matrix of the right type
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
    else if (type == GxB_FC64)
    { 
        return (mxCreateNumericMatrix (nrows, ncols, mxDOUBLE_CLASS,
            mxCOMPLEX)) ;
    }

    ERROR ("unsupported type", GrB_DOMAIN_MISMATCH) ;
    return (NULL) ;
}

