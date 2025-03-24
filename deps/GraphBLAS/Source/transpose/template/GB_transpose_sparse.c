//------------------------------------------------------------------------------
// GB_transpose_sparse: C=op(cast(A')), transpose, typecast, and apply op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //----------------------------------------------------------------------
    // A is sparse or hypersparse; C is sparse
    //----------------------------------------------------------------------

    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;
    ASSERT (GB_IS_SPARSE (C)) ;

    if (GB_Cp_IS_32)
    { 
        #define GB_W_TYPE uint32_t
        #define GB_ATOMIC_CAPTURE_INC(r,t) GB_ATOMIC_CAPTURE_INC32(r,t)
        #include "template/GB_transpose_sparse_template.c"
    }
    else
    {
        #define GB_W_TYPE uint64_t
        #define GB_ATOMIC_CAPTURE_INC(r,t) GB_ATOMIC_CAPTURE_INC64(r,t)
        #include "template/GB_transpose_sparse_template.c"
    }
}

