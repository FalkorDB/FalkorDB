//------------------------------------------------------------------------------
// GB_mx_Type: get GraphBLAS type of a built-in matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Given a built-in matrix, return the equivalent GraphBLAS type.  For GxB_FC64,
// the Complex type is returned.  This may equal GxB_FC64, or it might be the
// user-defined type, as determined by Complex_init in GB_mx_usercomplex.

#include "GB_mex.h"

GrB_Type GB_mx_Type                    // returns a GraphBLAS type
(
    const mxArray *X                   // built-in matrix to query
)
{

    GrB_Type xtype ;

    if (X == NULL)
    {
        return (NULL) ;
    }

    if (mxIsComplex (X))
    {
        switch (mxGetClassID (X))
        {
            // only single complex and double complex are supported
            case mxSINGLE_CLASS   : return (GxB_FC32  ) ;
            case mxDOUBLE_CLASS   : return (Complex   ) ;
            default               : return (NULL      ) ;
        }
    }
    else
    {
        switch (mxGetClassID (X))
        {
            // all GraphBLAS built-in types are supported
            case mxLOGICAL_CLASS  : return (GrB_BOOL  ) ;
            case mxINT8_CLASS     : return (GrB_INT8  ) ;
            case mxINT16_CLASS    : return (GrB_INT16 ) ;
            case mxINT32_CLASS    : return (GrB_INT32 ) ;
            case mxINT64_CLASS    : return (GrB_INT64 ) ;
            case mxUINT8_CLASS    : return (GrB_UINT8 ) ;
            case mxUINT16_CLASS   : return (GrB_UINT16) ;
            case mxUINT32_CLASS   : return (GrB_UINT32) ;
            case mxUINT64_CLASS   : return (GrB_UINT64) ;
            case mxSINGLE_CLASS   : return (GrB_FP32  ) ;
            case mxDOUBLE_CLASS   : return (GrB_FP64  ) ;
            default               : return (NULL      ) ;
        }
    }
}

