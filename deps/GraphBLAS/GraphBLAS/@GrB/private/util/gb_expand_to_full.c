//------------------------------------------------------------------------------
// gb_expand_to_full: add identity values to a matrix so all entries are present
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

GrB_Matrix gb_expand_to_full    // C = full (A), and typecast
(
    const GrB_Matrix A,         // input matrix to expand to full
    GrB_Type type,              // type of C, if NULL use the type of A
    GxB_Format_Value fmt,       // format of C
    GrB_Matrix id               // identity value, use zero if NULL
)
{

    //--------------------------------------------------------------------------
    // get the size and type of A
    //--------------------------------------------------------------------------

    GrB_Type atype ;
    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;
    OK (GxB_Matrix_type (&atype, A)) ;

    // C defaults to the same type of A
    if (type == NULL)
    {
        type = atype ;
    }

    //--------------------------------------------------------------------------
    // get the identity, use full(0) if NULL
    //--------------------------------------------------------------------------

    GrB_Matrix id2 = NULL ;
    if (id == NULL)
    {
        OK (GrB_Matrix_new (&id2, type, 1, 1)) ;
        OK (GrB_Matrix_setElement_INT32 (id2, 0, 0, 0)) ;
        id = id2 ;
    }

    //--------------------------------------------------------------------------
    // expand the identity into a full matrix B the same size as C
    //--------------------------------------------------------------------------

    GrB_Matrix B = gb_new (type, nrows, ncols, fmt, 0) ;
    OK1 (B, GrB_Matrix_assign_Scalar (B, NULL, NULL, (GrB_Scalar) id,
        GrB_ALL, 0, GrB_ALL, 0, NULL)) ;

    //--------------------------------------------------------------------------
    // typecast A from float to integer using the built-in rules
    //--------------------------------------------------------------------------

    GrB_Matrix S, T = NULL ;
    if (gb_is_integer (type) && gb_is_float (atype))
    { 
        // T = (type) round (A)
        T = gb_new (type, nrows, ncols, fmt, 0) ;
        OK1 (T, GrB_Matrix_apply (T, NULL, NULL, gb_round_op (atype), A, NULL));
        S = T ;
    }
    else
    { 
        // T = A, and let GrB_Matrix_eWiseAdd_BinaryOp do the typecasting
        S = A ;
    }

    //--------------------------------------------------------------------------
    // C = first (S, B)
    //--------------------------------------------------------------------------

    GrB_Matrix C = gb_new (type, nrows, ncols, fmt, 0) ;
    OK1 (C, GrB_Matrix_eWiseAdd_BinaryOp (C, NULL, NULL,
        gb_first_binop (type), S, B, NULL)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&id2)) ;
    OK (GrB_Matrix_free (&B)) ;
    OK (GrB_Matrix_free (&T)) ;
    return (C) ;
}

