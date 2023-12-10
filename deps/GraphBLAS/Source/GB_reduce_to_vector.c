//------------------------------------------------------------------------------
// GB_reduce_to_vector: reduce a matrix to a vector using a monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<M> = accum (C,reduce(A)) where C is n-by-1.  Reduces a matrix A or A'
// to a vector.

#define GB_FREE_ALL                     \
{                                       \
    GB_Matrix_free (&B) ;               \
    GrB_Semiring_free (&semiring) ;     \
}

#include "GB_reduce.h"
#include "GB_binop.h"
#include "GB_mxm.h"
#include "GB_get_mask.h"
#include "GB_Semiring_new.h"

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

    struct GB_Matrix_opaque B_header ;
    GrB_Matrix B = NULL ;
    struct GB_Semiring_opaque semiring_header ;
    GrB_Semiring semiring = NULL ;

    // C may be aliased with M and/or A
    GB_RETURN_IF_NULL_OR_FAULTY (C) ;
    GB_RETURN_IF_FAULTY (M_in) ;
    GB_RETURN_IF_FAULTY_OR_POSITIONAL (accum) ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    GB_RETURN_IF_FAULTY (desc) ;

    ASSERT_MATRIX_OK (C, "C input for reduce-to-vector", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M_in, "M_in for reduce-to-vector", GB0) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum for reduce-to-vector", GB0) ;
    ASSERT_MONOID_OK (monoid, "monoid for reduce-to-vector", GB0) ;
    ASSERT_MATRIX_OK (A, "A input for reduce-to-vector", GB0) ;
    ASSERT_DESCRIPTOR_OK_OR_NULL (desc, "desc for reduce-to-vector", GB0) ;
    ASSERT (GB_VECTOR_OK (C)) ;
    ASSERT (GB_IMPLIES (M_in != NULL, GB_VECTOR_OK (M_in))) ;

    // get the descriptor
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

    // B is constructed with a static header in O(1) time and space, even
    // though it is m-by-1.  It contains no dynamically-allocated content and
    // does not need to be freed.
    int64_t m = A_transpose ? GB_NROWS (A) : GB_NCOLS (A) ;
    GB_CLEAR_STATIC_HEADER (B, &B_header) ;
    info = GB_new (&B, // full, existing header
        ztype, m, 1, GB_Ap_null, true, GxB_FULL, GB_NEVER_HYPER, 1) ;
    ASSERT (info == GrB_SUCCESS) ;
    B->magic = GB_MAGIC ;
    B->iso = true ;             // OK: B is a temporary matrix; no burble
    size_t zsize = ztype->size ;
    GB_void bscalar [GB_VLA(zsize)] ;
    memset (bscalar, 0, zsize) ;
    B->x = bscalar ;
    B->x_shallow = true ;
    B->x_size = zsize ;
    ASSERT_MATRIX_OK (B, "B for reduce-to-vector", GB0) ;

    //--------------------------------------------------------------------------
    // create the FIRST_ZTYPE binary operator
    //--------------------------------------------------------------------------

    struct GB_BinaryOp_opaque op_header ;
    GrB_BinaryOp op ;

    switch (ztype->code)
    {
        case GB_BOOL_code   : op = GrB_FIRST_BOOL   ; break ;
        case GB_INT8_code   : op = GrB_FIRST_INT8   ; break ;
        case GB_INT16_code  : op = GrB_FIRST_INT16  ; break ;
        case GB_INT32_code  : op = GrB_FIRST_INT32  ; break ;
        case GB_INT64_code  : op = GrB_FIRST_INT64  ; break ;
        case GB_UINT8_code  : op = GrB_FIRST_UINT8  ; break ;
        case GB_UINT16_code : op = GrB_FIRST_UINT16 ; break ;
        case GB_UINT32_code : op = GrB_FIRST_UINT32 ; break ;
        case GB_UINT64_code : op = GrB_FIRST_UINT64 ; break ;
        case GB_FP32_code   : op = GrB_FIRST_FP32   ; break ;
        case GB_FP64_code   : op = GrB_FIRST_FP64   ; break ;
        case GB_FC32_code   : op = GxB_FIRST_FC32   ; break ;
        case GB_FC64_code   : op = GxB_FIRST_FC64   ; break ;
        default : 
            // Create a FIRST_UDT binary operator.  The function pointer for
            // the FIRST_UDT op is NULL; it is not needed by FIRST.  The
            // function defn is also NULL.  In the JIT, the FIRST multiply
            // operator is a simple assignment so there's no need for a
            // function definition.  This binary op will not be treated as a
            // builtin operator, however, since its data type is not builtin.
            // Its hash, op->hash, will be nonzero.  The name of FIRST_UDT is
            // just "1st", which is not itself unique, but it will only be used
            // in combination with the monoid, which must be user-defined.  No
            // typecasting is allowed between user-defined types, so the
            // user-defined monoid must be specific this particular
            // user-defined type and this "reduce_1st" will be a unique name
            // for the constructed semiring (if "reduce" is the name of the
            // monoid).  In addition, it is not possible for the user to create
            // a jitfyable operator with the name "1st", because of the leading
            // "1" character in its name.  So "reduce_1st" must be unique.
            op = &op_header ;
            op->header_size = 0 ;
            info = GB_binop_new (op,
                NULL,   // op->binop_function is NULL for FIRST_UDT
                ztype, ztype, ztype,    // ztype is user-defined
                "1st",                  // a simple name for FIRST_UDT
                NULL,   // no op->defn for the FIRST_UDT operator
                GB_FIRST_binop_code) ;  // using a built-in opcode
            break ;
    }

    // GB_binop_new cannot fail since it doesn't allocate the function defn.
    ASSERT (info == GrB_SUCCESS) ;
    ASSERT_BINARYOP_OK (op, "op for reduce-to-vector", GB0) ;

    //--------------------------------------------------------------------------
    // create the REDUCE_FIRST_ZTYPE semiring
    //--------------------------------------------------------------------------

    semiring = &semiring_header ;
    semiring->header_size = 0 ;
    info = GB_Semiring_new (semiring, monoid, op) ;
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

