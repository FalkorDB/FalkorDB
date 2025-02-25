//------------------------------------------------------------------------------
// gb_new: create a GraphBLAS matrix with desired format and sparsity control
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

GrB_Matrix gb_new       // create and empty matrix C
(
    GrB_Type type,      // type of C
    uint64_t nrows,     // # of rows
    uint64_t ncols,     // # of rows
    int fmt,            // requested format, if < 0 use default
    int sparsity        // sparsity control for C, 0 for default
)
{

    // create the matrix
    GrB_Matrix C = NULL ;
    OK (GrB_Matrix_new (&C, type, nrows, ncols)) ;

    // get the default format, if needed
    if (fmt < 0)
    { 
        fmt = gb_default_format (nrows, ncols) ;
    }

    // set the desired format
    int fmt_current ;
    OK (GrB_Matrix_get_INT32 (C, &fmt_current, GxB_FORMAT)) ;
    if (fmt != fmt_current)
    { 
        OK (GrB_Matrix_set_INT32 (C, fmt, GxB_FORMAT)) ;
    }

    // set the desired sparsity structure
    if (sparsity != 0)
    { 
        int current ;
        OK (GrB_Matrix_get_INT32 (C, &current, GxB_SPARSITY_CONTROL)) ;
        if (current != sparsity)
        {
            OK (GrB_Matrix_set_INT32 (C, sparsity, GxB_SPARSITY_CONTROL)) ;
        }
    }

    return (C) ;
}

