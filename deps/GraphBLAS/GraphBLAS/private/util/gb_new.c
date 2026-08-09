//------------------------------------------------------------------------------
// gb_new: create a GraphBLAS matrix with desired format and sparsity control
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#undef  FREE_ALL
#define FREE_ALL GrB_Matrix_free (&C) ;

GrB_Info gb_new       // create and empty matrix C
(
    // output
    GrB_Matrix *C_handle,
    // input
    GrB_Type type,      // type of C
    uint64_t nrows,     // # of rows
    uint64_t ncols,     // # of rows
    int fmt,            // requested format, if < 0 use default
    int sparsity,       // sparsity control for C, 0 for default
    int arena,
    char err [ERRLEN]
)
{

    // create the matrix
    GrB_Matrix C = NULL ;
    OK (GxB_Matrix_new_arena (&C, type, nrows, ncols, arena, arena)) ;

    // get the default format, if needed
    if (fmt < 0)
    { 
        OK (gb_default_format (&fmt, nrows, ncols, err)) ;
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

    (*C_handle) = C ;
    return (GrB_SUCCESS) ;
}

#undef  FREE_ALL
#define FREE_ALL

