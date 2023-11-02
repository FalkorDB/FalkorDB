//------------------------------------------------------------------------------
// gbtype: type of a GraphBLAS matrix struct, or any built-in variable
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input may be any built-in variable.  If it is a GraphBLAS G.opaque
// struct, then its internal type is returned.

// Usage

// type = gbtype (X)

#include "gb_interface.h"

#define USAGE "usage: type = gbtype (X)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin == 1 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // get the type of the matrix
    //--------------------------------------------------------------------------

    mxArray *c = NULL ;
    mxClassID class = mxGetClassID (pargin [0]) ;
    bool is_complex = mxIsComplex (pargin [0]) ;

    if (class == mxSTRUCT_CLASS)
    {
        // get the content of a GraphBLASv7_3 struct
        mxArray *mx_type = mxGetField (pargin [0], 0, "GraphBLASv7_3") ;
        if (mx_type == NULL)
        { 
            // check if it is a GraphBLASv5_1 struct
            mx_type = mxGetField (pargin [0], 0, "GraphBLASv5_1") ;
        }
        if (mx_type == NULL)
        { 
            // check if it is a GraphBLASv5 struct
            mx_type = mxGetField (pargin [0], 0, "GraphBLASv5") ;
        }
        if (mx_type == NULL)
        { 
            // check if it is a GraphBLASv4 struct
            mx_type = mxGetField (pargin [0], 0, "GraphBLASv4") ;
        }
        if (mx_type == NULL)
        { 
            // check if it is a GraphBLASv3 struct
            mx_type = mxGetField (pargin [0], 0, "GraphBLAS") ;
        }
        if (mx_type != NULL)
        {
            // the mxArray is a struct containing a GraphBLAS GrB_matrix;
            // get its type
            c = mxDuplicateArray (mx_type) ;
        }
    }

    if (c == NULL)
    { 
        // if c is still NULL, then it is not a GraphBLAS opaque struct.
        // get the type of a built-in matrix
        c = gb_mxclass_to_mxstring (class, is_complex) ;
    }

    //--------------------------------------------------------------------------
    // return the result
    //--------------------------------------------------------------------------

    pargout [0] = c ;
    GB_WRAPUP ;
}

