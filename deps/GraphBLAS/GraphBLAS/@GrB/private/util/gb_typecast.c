//------------------------------------------------------------------------------
// gb_typecast: typecast a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

GrB_Matrix gb_typecast          // C = (type) A, where C is deep
(
    GrB_Matrix A,               // may be shallow
    GrB_Type type,              // if NULL, use the type of A
    GxB_Format_Value fmt,       // format of C
    int sparsity                // sparsity control for C, if 0 use A
)
{

    //--------------------------------------------------------------------------
    // determine the sparsity control for C
    //--------------------------------------------------------------------------

    sparsity = gb_get_sparsity (A, NULL, sparsity) ;

    //--------------------------------------------------------------------------
    // get the type of C and A
    //--------------------------------------------------------------------------

    GrB_Type atype ;
    OK (GxB_Matrix_type (&atype, A)) ;
    if (type == NULL)
    { 
        // keep the same type
        type = atype ;
    }

    //--------------------------------------------------------------------------
    // create the empty C matrix and set its format and sparsity
    //--------------------------------------------------------------------------

    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;
    GrB_Matrix C = gb_new (type, nrows, ncols, fmt, sparsity) ;

    //--------------------------------------------------------------------------
    // C = A
    //--------------------------------------------------------------------------

    if (gb_is_integer (type) && gb_is_float (atype))
    { 
        // C = (type) round (A), using built-in rules for typecasting.
        OK1 (C, GrB_Matrix_apply (C, NULL, NULL, gb_round_op (atype), A, NULL));
    }
    else
    { 
        // C = (type) A, with GraphBLAS typecasting if needed.
        OK1 (C, GrB_Matrix_assign (C, NULL, NULL, A,
            GrB_ALL, nrows, GrB_ALL, ncols, NULL)) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (C) ;
}

