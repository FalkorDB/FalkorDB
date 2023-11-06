//------------------------------------------------------------------------------
// gbdisp: display a GraphBLAS matrix struct
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbdisp (C, cnz, level)

#include "gb_interface.h"

#define USAGE "usage: gbdisp (C, cnz, level)"

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

    gb_usage (nargin == 3 && nargout == 0, USAGE) ;

    //--------------------------------------------------------------------------
    // get cnz and level
    //--------------------------------------------------------------------------

    double cnz = mxGetScalar (pargin [1]) ;
    int level = (int) mxGetScalar (pargin [2]) ;

    #define LEN 256
    char s [LEN+1] ;
    if (cnz == 0)
    { 
        snprintf (s, LEN, "no nonzeros") ;
    }
    else if (cnz == 1)
    { 
        snprintf (s, LEN, "1 nonzero") ;
    }
    else if (cnz < INT64_MAX)
    {
        snprintf (s, LEN, GBd " nonzeros", (int64_t) cnz) ;
    }
    else
    { 
        snprintf (s, LEN, "%g nonzeros", cnz) ;
    }

    s [LEN] = '\0' ;

    //--------------------------------------------------------------------------
    // print the GraphBLAS matrix
    //--------------------------------------------------------------------------

    // print 1-based indices
    OK (GxB_Global_Option_set (GxB_PRINT_1BASED, true)) ;

    // print sizes of shallow components
    GB_Global_print_mem_shallow_set (true) ;

    GrB_Matrix C = gb_get_shallow (pargin [0]) ;
    OK (GxB_Matrix_fprint (C, s, level, NULL)) ;
    OK (GrB_Matrix_free (&C)) ;
    GB_WRAPUP ;
}

