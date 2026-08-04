//------------------------------------------------------------------------------
// gb_expand_to_full: add identity values to a matrix so all entries are present
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#undef  FREE_WORK
#define FREE_WORK               \
    GrB_Matrix_free (&id2) ;    \
    GrB_Matrix_free (&B) ;      \
    GrB_Matrix_free (&T) ;

#undef  FREE_ALL
#define FREE_ALL                \
    FREE_WORK ;                 \
    GrB_Matrix_free (&C) ; 

GrB_Info gb_expand_to_full      // C = full (A), and typecast
(
    // output
    GrB_Matrix *C_handle,
    // inputs
    const GrB_Matrix A,         // input matrix to expand to full
    GrB_Type type,              // type of C, if NULL use the type of A
    int fmt,                    // format of C
    GrB_Matrix id,              // identity value, use zero if NULL
    const int arena,
    char err [ERRLEN]
)
{

    GrB_Matrix C = NULL, id2 = NULL, B = NULL, T = NULL, S = NULL ;

    //--------------------------------------------------------------------------
    // get the size and type of A
    //--------------------------------------------------------------------------

    GrB_Type atype ;
    uint64_t nrows, ncols ;
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

    if (id == NULL)
    { 
        OK (GxB_Matrix_new_arena (&id2, type, 1, 1, arena, arena)) ;
        OK (GrB_Matrix_setElement_INT32 (id2, 0, 0, 0)) ;
        id = id2 ;
    }

    //--------------------------------------------------------------------------
    // expand the identity into a full matrix B the same size as C
    //--------------------------------------------------------------------------

    OK (gb_new (&B, type, nrows, ncols, fmt, 0, arena, err)) ;
    OK1 (B, GrB_Matrix_assign_Scalar (B, NULL, NULL, (GrB_Scalar) id,
        GrB_ALL, 0, GrB_ALL, 0, NULL)) ;

    //--------------------------------------------------------------------------
    // typecast A from float to integer using the built-in rules
    //--------------------------------------------------------------------------

    if (gb_is_integer (type) && gb_is_float (atype))
    { 
        // T = (type) round (A)
        OK (gb_new (&T, type, nrows, ncols, fmt, 0, arena, err)) ;
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

    GrB_BinaryOp op ;
    OK (gb_new (&C, type, nrows, ncols, fmt, 0, arena, err)) ;
    OK (gb_first_binop (&op, type, err)) ;
    OK1 (C, GrB_Matrix_eWiseAdd_BinaryOp (C, NULL, NULL, op, S, B, NULL)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    (*C_handle) = C ;
    return (GrB_SUCCESS) ;
}

#undef  FREE_WORK
#define FREE_WORK
#undef  FREE_ALL
#define FREE_ALL

