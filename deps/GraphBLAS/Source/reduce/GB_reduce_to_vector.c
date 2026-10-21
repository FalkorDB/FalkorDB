//------------------------------------------------------------------------------
// GB_reduce_to_vector: reduce a matrix to a vector using a monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<M> = accum (C,reduce(A)) where C is n-by-1.  Reduces a matrix A or A'
// to a vector.

#define GB_FREE_ALL                     \
{                                       \
    GB_Matrix_free (&B) ;               \
    GrB_Semiring_free (&semiring) ;     \
}

#include "reduce/GB_reduce.h"
#include "binaryop/GB_binop.h"
#include "mxm/GB_mxm.h"
#include "mask/GB_get_mask.h"
#include "semiring/GB_Semiring_new.h"

GrB_Info GB_reduce_to_vector        // C<M> = accum (C,reduce(A))
(
    GrB_Matrix C,                   // input/output for results, size n-by-1
    const GrB_Matrix M_in,          // optional M for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(C,T)
    const GrB_Monoid monoid,        // reduce monoid for T=reduce(A)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Descriptor desc,      // descriptor for C, M, and A
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_FAULTY_OR_POSITIONAL (accum) ;
    GB_RETURN_IF_FAULTY (desc) ;

    GrB_Matrix B = NULL ;
    struct GB_Semiring_opaque semiring_header ;
    GrB_Semiring semiring = NULL ;

    ASSERT_MATRIX_OK (C, "C input for reduce-to-vector", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M_in, "M_in for reduce-to-vector", GB0) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum for reduce-to-vector", GB0) ;
    ASSERT_MONOID_OK (monoid, "monoid for reduce-to-vector", GB0) ;
    ASSERT_MATRIX_OK (A, "A input for reduce-to-vector", GB0) ;
    ASSERT_DESCRIPTOR_OK_OR_NULL (desc, "desc for reduce-to-vector", GB0) ;
    ASSERT (GB_VECTOR_OK (C)) ;
    ASSERT (GB_IMPLIES (M_in != NULL, GB_VECTOR_OK (M_in))) ;

    int data_arena = C->data_arena ;

    // get the descriptor
    GrB_Info info ;
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        A_transpose, xx1, xx2, do_sort) ;

    // get the mask
    GrB_Matrix M = GB_get_mask (M_in, &Mask_comp, &Mask_struct) ;

    // check domains and dimensions for C<M> = accum (C,T)
    GrB_Type ztype = monoid->op->ztype ;
    GB_OK (GB_compatible (C->type, C, M, Mask_struct, accum, ztype, Werk)) ;

    // T = reduce (T,A) must be compatible
    if (!GB_Type_compatible (A->type, ztype))
    { 
        GB_ERROR (GrB_DOMAIN_MISMATCH,
            "Incompatible type for reduction monoid z=%s(x,y):\n"
            "input matrix A of type [%s]\n"
            "cannot be typecast to reduction monoid of type [%s]",
            monoid->op->name, A->type->name, ztype->name) ;
    }

    // check the dimensions
    int64_t n = GB_NROWS (C) ;
    if (A_transpose)
    {
        if (n != GB_NCOLS (A))
        { 
            GB_ERROR (GrB_DIMENSION_MISMATCH,
                "w=reduce(A'):  length of w is " GBd ";\n"
                "it must match the number of columns of A, which is " GBd ".",
                n, GB_NCOLS (A)) ;
        }
    }
    else
    {
        if (n != GB_NROWS(A))
        { 
            GB_ERROR (GrB_DIMENSION_MISMATCH,
                "w=reduce(A):  length of w is " GBd ";\n"
                "it must match the number of rows of A, which is " GBd ".",
                n, GB_NROWS (A)) ;
        }
    }

    // quick return if an empty mask is complemented
    GB_RETURN_IF_QUICK_MASK (C, C_replace, M, Mask_comp, Mask_struct) ;

    //--------------------------------------------------------------------------
    // create B as full iso vector
    //--------------------------------------------------------------------------

    // B is constructed in O(1) time and space, even though it is m-by-1.
    int64_t m = A_transpose ? GB_NROWS (A) : GB_NCOLS (A) ;
    GB_OK (GB_new_bix (&B, // full, new header
        /* type: */ ztype, /* vlen, vdim: */ m, 1, 
        /* Ap_option: */ GB_ph_null, /* is_csc: */ true,
        /* sparsity: */ GxB_FULL, /* bitmap_calloc unused: */ false,
        /* hyper_switch: */ GB_NEVER_HYPER, /* plen: */ 1,
        /* nzmax: */ 1, /* numeric: */ true, /* iso: */ true,
        /* pji_is_32: */ false, false, false,
        /* header and data arena: */ data_arena, data_arena)) ;
    ASSERT (B->iso) ;
    ASSERT (!B->x_shallow) ;
    B->magic = GB_MAGIC ;
    memset (B->x, 0, ztype->size) ;
    ASSERT_MATRIX_OK (B, "B for reduce-to-vector", GB0) ;

    //--------------------------------------------------------------------------
    // create the FIRST_ZTYPE binary operator
    //--------------------------------------------------------------------------

    struct GB_BinaryOp_opaque op_header ;
    GrB_BinaryOp op = GB_binop_first (ztype, &op_header) ;
    ASSERT_BINARYOP_OK (op, "op for reduce-to-vector", GB0) ;

    //--------------------------------------------------------------------------
    // create the REDUCE_FIRST_ZTYPE semiring
    //--------------------------------------------------------------------------

    semiring = &semiring_header ;
    semiring->header_mem = 0 ;  // static header for semiring
    info = GB_Semiring_new (semiring, monoid, op, data_arena) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        // GB_Semiring_new allocates semiring->name if it uses the FIRST_UDT
        // operator created above, so it can run out of memory in that case.
        GB_FREE_ALL ;
        return (info) ;
    }

    ASSERT_SEMIRING_OK (semiring, "semiring for reduce-to-vector", GB0) ;

    if (GB_Global_burble_get ( ))
    { 
        GB_Semiring_check (semiring, "semiring for reduce-to-vector", 3, NULL) ;
    }

    //--------------------------------------------------------------------------
    // reduce the matrix to a vector via C<M> = accum (C, A*B)
    //--------------------------------------------------------------------------

    info = GB_mxm (C, C_replace, M, Mask_comp, Mask_struct, accum,
        semiring, A, A_transpose, B, false, false, GxB_DEFAULT, do_sort, Werk) ;
    GB_FREE_ALL ;
    return (info) ;
}

