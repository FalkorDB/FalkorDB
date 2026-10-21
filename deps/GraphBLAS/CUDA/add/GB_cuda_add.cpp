//------------------------------------------------------------------------------
// CUDA/apply/GB_cuda_add: C=A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#undef  GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE                                   \
{                                                           \
    GB_FREE_MEMORY (&alpha_scalar, alpha_scalar_mem) ;      \
    GB_FREE_MEMORY (&beta_scalar , beta_scalar_mem ) ;      \
    GB_FREE_MEMORY (&theta, theta_mem) ;                    \
    GB_cuda_stream_pool_release (&stream) ;                 \
    GB_Matrix_free (&A2) ;                                  \
    GB_Matrix_free (&B2) ;                                  \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL                                         \
{                                                           \
    GB_FREE_WORKSPACE ;                                     \
    GB_phybix_free (C) ;                                    \
}

#include "add/GB_cuda_add.hpp"
#include "mask/GB_cuda_mask.hpp"

GrB_Info GB_cuda_add
(
    // output:
    GrB_Matrix C,               // existing header with no content
    // inputs:
    const GrB_Type ctype,       // type of C
    const bool C_is_csc,        // true if C is held by column, false if by row
    const GrB_Matrix M,         // input mask matrix
    const bool Mask_struct,     // if true, mask is structural
    const bool Mask_comp,       // if true, mask is complemented
    const GrB_Matrix A,
    const GrB_Matrix B,
    const bool is_eWiseUnion,   // true if eWiseUnion, false if eWiseAdd
    const GrB_Scalar alpha,     // alpha for eWiseUnion
    const GrB_Scalar beta,      // beta for eWiseUnion
    const GrB_BinaryOp op,      // operator that defines C=A+B
    const bool flipij,          // true if i and j are reversed in the op
    const bool A_and_B_are_disjoint,    // true if A and B are known to be
                                        // disjoint (for GrB_wait only)
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    // FIXME: valued mask not yet handled
    ASSERT (GB_IMPLIES (M != NULL, Mask_struct)) ;

    GrB_Info info ;
    cudaStream_t stream = nullptr ;
    GrB_Matrix A2 = NULL, B2 = NULL ;

    int data_arena = C->data_arena ;
    int device = data_arena - GxB_NARENAS ;
    uint64_t mem = GB_mem (data_arena, 0) ;

    GB_void *alpha_scalar = NULL ; uint64_t alpha_scalar_mem = mem ;
    GB_void *beta_scalar  = NULL ; uint64_t beta_scalar_mem  = mem ;
    void *theta           = NULL ; uint64_t theta_mem        = mem ;

    GB_OK (GB_cuda_stream_pool_acquire (device, &stream)) ;
    GB_OK (GB_wait_arenas (A)) ;
    GB_OK (GB_wait_arenas (B)) ;

    //--------------------------------------------------------------------------
    // typecast alpha and beta for eWiseUnion
    //--------------------------------------------------------------------------

    if (is_eWiseUnion)
    {
        alpha_scalar = (GB_void *) GB_MALLOC_MEMORY (1, op->xtype->size,
            &alpha_scalar_mem);
        beta_scalar  = (GB_void *) GB_MALLOC_MEMORY (1, op->ytype->size,
            &beta_scalar_mem );
        if (alpha_scalar == NULL || beta_scalar == NULL)
        {
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        // alpha_scalar = (xtype) alpha
        ASSERT (alpha != NULL) ;
        GB_cast_scalar (alpha_scalar, op->xtype->code, alpha->x,
            alpha->type->code, alpha->type->size) ;

        // beta_scalar = (ytype) beta
        ASSERT (beta != NULL) ;
        GB_cast_scalar (beta_scalar, op->ytype->code, beta->x,
            beta->type->code, beta->type->size) ;
    }

    //--------------------------------------------------------------------------
    // copy theta scalar for the GPU, for index binary ops only
    //--------------------------------------------------------------------------

    if (op->theta != NULL)
    {
        theta = GB_MALLOC_MEMORY (1, op->theta_type->size , &theta_mem) ;
        if (theta == NULL)
        {
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
        memcpy (theta, op->theta, op->theta_type->size ) ;
    }

    //--------------------------------------------------------------------------
    // check if C is iso and compute its iso value if it is
    //--------------------------------------------------------------------------

    size_t csize = ctype->size ;
    GB_void cscalar [GB_VLA(csize)] ;
    bool C_iso = GB_add_iso (cscalar, ctype, A, alpha_scalar,
        B, beta_scalar, op, A_and_B_are_disjoint, is_eWiseUnion) ;

    //--------------------------------------------------------------------------
    // define the header of C
    //--------------------------------------------------------------------------

    uint64_t anz = GB_nnz_held (A) ;
    uint64_t bnz = GB_nnz_held (B) ;
    uint64_t cnz = anz+bnz ;            // upper bound

    bool Cp_is_32, Cj_is_32, Ci_is_32 ;
    GB_determine_pji_is_32 (&Cp_is_32, &Cj_is_32, &Ci_is_32,
        GxB_HYPERSPARSE, cnz, A->vlen, A->vdim, Werk) ;

    // initialize the header of C, with no phybix content,
    // and initialize the type and dimension of C.
    info = GB_new (&C, // hyper, existing header
        ctype, A->vlen, A->vdim, GB_ph_null, C_is_csc,
        GxB_HYPERSPARSE, GB_Global_hyper_switch_get ( ), 0,
        Cp_is_32, Cj_is_32, Ci_is_32, data_arena, data_arena) ;
    ASSERT (info == GrB_SUCCESS) ;

    //--------------------------------------------------------------------------
    // C=A+B on CUDA, in the JIT
    //--------------------------------------------------------------------------

    int32_t number_of_sms = GB_Global_gpu_sm_get (device) ;
    int64_t raw_gridsz = GB_ICEIL (cnz, GB_CUDA_ADD_CHUNKSIZE0) ;
    // cap #of blocks to 256 * #of sms
    int32_t gridsz = std::min (raw_gridsz, (int64_t) (number_of_sms * 256)) ;

    GBURBLE ("(cuda add, device %d, sms: %d, gridsz: %d) ",
        device, number_of_sms, gridsz) ;

    if (M == NULL)
    {

        //----------------------------------------------------------------------
        // C = A+B
        //----------------------------------------------------------------------

        GB_OK (GB_cuda_add_jit (C, C_iso, A, B, is_eWiseUnion,
            alpha_scalar, beta_scalar, theta, op, flipij,
            A_and_B_are_disjoint, device, stream, gridsz)) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // C<M> = A+B or C<!M> = A+B
        //----------------------------------------------------------------------

        GBURBLE ("(cuda masked add, Mask comp: %d) ", Mask_comp) ;

        GB_OK (GB_new (&A2, // hyper, new header
            A->type, A->vlen, A->vdim, GB_ph_null, A->is_csc,
            GxB_HYPERSPARSE, GB_Global_hyper_switch_get ( ), 0,
            A->p_is_32, A->j_is_32, A->i_is_32, data_arena, data_arena)) ;

        GB_OK (GB_new (&B2, // hyper, new header
            B->type, B->vlen, B->vdim, GB_ph_null, B->is_csc,
            GxB_HYPERSPARSE, GB_Global_hyper_switch_get ( ), 0,
            B->p_is_32, B->j_is_32, B->i_is_32, data_arena, data_arena)) ;

        // A2 <#M> = A
        printf ("\nCUDA A2<%dM>=A:\n", Mask_comp) ;
        struct GB_BinaryOp_opaque op_header ;
        GrB_BinaryOp first = GB_binop_first (A->type, &op_header) ;
        ASSERT_BINARYOP_OK (first, "first_atype for A2<*M>=A", GB0) ;
        GB_OK (GB_cuda_mask_assign_jit (A2, A, M, Mask_struct, Mask_comp,
            first, device, stream, gridsz)) ;

//      ASSERT_MATRIX_OK (A, "A input for masking", GB0) ;
//      ASSERT_MATRIX_OK (M, "M for masking", GB0) ;
//      ASSERT_MATRIX_OK (A2, "A2 output from masking", GB0) ;

        // B2 <#M> = A
        printf ("\nCUDA B2<%dM>=B:\n", Mask_comp) ;
        first = GB_binop_first (B->type, &op_header) ;
        ASSERT_BINARYOP_OK (first, "first_btype for B2<*M>=B", GB0) ;
        GB_OK (GB_cuda_mask_assign_jit (B2, B, M, Mask_struct, Mask_comp,
            first, device, stream, gridsz)) ;

//      ASSERT_MATRIX_OK (B, "A input for masking", GB0) ;
//      ASSERT_MATRIX_OK (M, "M for masking", GB0) ;
//      ASSERT_MATRIX_OK (B2, "A2 output from masking", GB0) ;

        // C = A2 + B2
        printf ("\nCUDA C=A2+B2\n") ;
        GB_OK (GB_cuda_add_jit (C, C_iso, A2, B2, is_eWiseUnion,
            alpha_scalar, beta_scalar, theta, op, flipij,
            A_and_B_are_disjoint, device, stream, gridsz)) ;
        printf ("\nCUDA C=A2+B2 done\n") ;
    }

    //--------------------------------------------------------------------------
    // set the iso value if C is iso
    //--------------------------------------------------------------------------

    if (C_iso)
    {
        GB_BURBLE_MATRIX (C, "(iso add) ") ;
        memcpy (C->x, cscalar, csize) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    return (GrB_SUCCESS) ;
}

