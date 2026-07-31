//------------------------------------------------------------------------------
// gb_default_format: determine the default format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GrB_Info gb_default_format
(
    // output
    int *fmt,               // GxB_BY_ROW or GxB_BY_COL
    // input
    uint64_t nrows,        // row vectors are stored by row
    uint64_t ncols,        // column vectors are stored by column
    char err [ERRLEN]
)
{

    if (ncols == 1)
    { 
        // column vectors are stored by column, by default
        (*fmt) = GxB_BY_COL ;
    }
    else if (nrows == 1)
    { 
        // row vectors are stored by row, by default
        (*fmt) = GxB_BY_ROW ;
    }
    else
    { 
        // get the default format
        OK (GrB_Global_get_INT32 (GrB_GLOBAL, fmt, GxB_FORMAT)) ;
    }
    return (GrB_SUCCESS) ;
}

