//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_AxB_dot3: compute C<M> = A'*B on GPU(s)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// This file: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This function computes C<M>=A'*B on the GPUs.  The mask must be present,
// sparse or hypersparse, and not complemented.  The mask is always applied.  A
// and B can have any sparsity format.  C is computed as sparse or hypersparse,
// with the same format as M.

#define GB_FREE_WORKSPACE                                               \
{                                                                       \
    /* FIXME: use a stream pool instead */                              \
    if (stream != nullptr) cudaStreamDestroy (stream) ;                 \
    stream = nullptr ;                                                  \
}

#define GB_FREE_ALL                                                     \
{                                                                       \
    GB_FREE_WORKSPACE ;                                                 \
    GB_phybix_free (C) ;                                                \
}

#include "GB_cuda_AxB.hpp"

//------------------------------------------------------------------------------
// GB_cuda_AxB_dot3
//------------------------------------------------------------------------------

GrB_Info GB_cuda_AxB_dot3           // C<M> = A'*B using dot product method
(
    GrB_Matrix C,                   // output matrix
    const GrB_Matrix M,             // mask matrix
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix A,             // input matrix
    const GrB_Matrix B,             // input matrix
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy               // if true, do z=fmult(b,a) vs fmult(a,b)
)
{

    cudaStream_t stream = nullptr ;

    //--------------------------------------------------------------------------
    // create the stream
    //--------------------------------------------------------------------------

    // FIXME: pass in a stream instead, or checkout a stream
    CUDA_OK (cudaStreamCreate (&stream)) ;
    GpuTimer kernel_timer;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // when CUDA is enabled, no static headers are used in all of GraphBLAS
    GrB_Info info ;
    ASSERT (C != NULL && !(C->static_header)) ;
    ASSERT (M != NULL && !(M->static_header)) ;
    ASSERT (A != NULL && !(A->static_header)) ;
    ASSERT (B != NULL && !(B->static_header)) ;

    ASSERT_MATRIX_OK (M, "M for dot3 cuda A'*B", GB0) ;
    ASSERT_MATRIX_OK (A, "A for dot3 cuda A'*B", GB0) ;
    ASSERT_MATRIX_OK (B, "B for dot3 cuda A'*B", GB0) ;

    ASSERT (!GB_PENDING (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (!GB_ZOMBIES (M)) ;

    ASSERT (!GB_PENDING (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;

    ASSERT (!GB_PENDING (B)) ;
    ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT (!GB_JUMBLED (B)) ;

    ASSERT_SEMIRING_OK (semiring, "semiring for dot3 numeric A'*B", GB0) ;

    ASSERT (A->vlen == B->vlen) ;
    GBURBLE ("(GPU dot3) ") ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    int device = -1;

    // FIXME: control the GPU to use via the descriptor
    CUDA_OK (cudaSetDevice ( 0 )) ;
    CUDA_OK (cudaGetDevice (&device)) ;
    int number_of_sms = GB_Global_gpu_sm_get (0) ;

    //--------------------------------------------------------------------------
    // get M, A, and B
    //--------------------------------------------------------------------------

    const int64_t mvlen = M->vlen ;
    const int64_t mvdim = M->vdim ;
    const int64_t mnz = GB_nnz (M) ;
    const int64_t mnvec = M->nvec ;
    const bool M_is_hyper = GB_IS_HYPERSPARSE( M ) ;

    const int64_t anz = GB_nnz (A) ;
    const int64_t anvec = A->nvec ;
    bool A_is_sparse = GB_IS_SPARSE (A) ;
    bool A_is_hyper  = GB_IS_HYPERSPARSE (A) ;
    bool A_is_bitmap = GB_IS_BITMAP (A) ;
    bool A_is_full   = GB_IS_FULL (A) ;
    bool A_is_sparse_or_hyper = A_is_sparse || A_is_hyper ;
    bool A_is_bitmap_or_full  = A_is_bitmap || A_is_full  ;

    const int64_t bnz = GB_nnz (B) ;
    const int64_t bnvec = B->nvec ;
    bool B_is_sparse = GB_IS_SPARSE (B) ;
    bool B_is_hyper  = GB_IS_HYPERSPARSE (B) ;
    bool B_is_bitmap = GB_IS_BITMAP (B) ;
    bool B_is_full   = GB_IS_FULL (B) ;
    bool B_is_sparse_or_hyper = B_is_sparse || B_is_hyper ;
    bool B_is_bitmap_or_full  = B_is_bitmap || B_is_full  ;

    //--------------------------------------------------------------------------
    // get the semiring operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp mult = semiring->multiply ;
    GrB_Monoid add = semiring->add ;
    ASSERT (mult->ztype == add->op->ztype) ;
    GB_Opcode mult_opcode = mult->opcode ;
    if (mult->xtype->code == GB_BOOL_code)
    {
        mult_opcode = GB_boolean_rename (mult_opcode) ;
    }
    bool A_is_pattern, B_is_pattern ;
    GB_binop_pattern (&A_is_pattern, &B_is_pattern, flipxy, mult_opcode) ;

    //--------------------------------------------------------------------------
    // allocate C, the same size and # of entries as M
    //--------------------------------------------------------------------------

    // FUTURE: ctype need not be the op->ztype
    GrB_Type ctype = add->op->ztype ;
    int64_t cvlen = mvlen ;
    int64_t cvdim = mvdim ;
    int64_t cnz = mnz ;
    int64_t cnvec = mnvec ;

    int M_sparsity = (M_is_hyper) ? GxB_HYPERSPARSE : GxB_SPARSE ;
    int C_sparsity = M_sparsity ;
    bool C_iso = false ;    // FIXME: pass in C_iso and cscalar
    bool C_in_iso = false ;    // FIXME: pass in C_in_iso and cscalar

    if (C_iso)
    {
        A_is_pattern = true ;
        B_is_pattern = true ;
    }

    GB_OK (GB_new_bix (&C, // sparse or hyper (from M), existing header
        ctype, cvlen, cvdim, GB_Ap_malloc, true,
        M_sparsity, false, M->hyper_switch, cnvec,
        cnz+1,  // add one to cnz for cumsum of Cwork
        true, C_iso)) ;

    //--------------------------------------------------------------------------
    // Pre-fetch arrays that will be used on the device
    //--------------------------------------------------------------------------

    // GB_cuda_matrix_advise (C, cnvec, cnz, which, what, device)
    // advise C
    CUDA_OK (cudaMemAdvise (C->p, (cnvec+1) * sizeof ( int64_t),
        cudaMemAdviseSetPreferredLocation, device)) ;
    if (M_is_hyper)
    { 
        CUDA_OK (cudaMemAdvise (C->h, cnvec * sizeof ( int64_t),
            cudaMemAdviseSetPreferredLocation, device)) ;
    }
    CUDA_OK (cudaMemAdvise (C->i, (cnz+1) * sizeof ( int64_t),
        cudaMemAdviseSetPreferredLocation, device)) ;
    if (!C_iso)
    {
        CUDA_OK (cudaMemAdvise (C->x, (cnz+1) * C->type->size ,
            cudaMemAdviseSetPreferredLocation, device)) ;
    }

    // prefetch M (if M hypersparse: using M->h not M->Y)
    GB_OK (GB_cuda_matrix_prefetch (M,
        Mask_struct ? GB_PREFETCH_PHBI : GB_PREFETCH_PHBIX, device, stream)) ;

    //--------------------------------------------------------------------------
    // copy Mp and Mh into C
    //--------------------------------------------------------------------------

    // FIXME: use shallow?
    CUDA_OK (cudaMemcpyAsync (C->p, M->p, (cnvec+1) * sizeof (int64_t),
        cudaMemcpyDefault, stream)) ;
    if (M_is_hyper)
    { 
        CUDA_OK (cudaMemcpyAsync (C->h, M->h, cnvec * sizeof (int64_t),
            cudaMemcpyDefault, stream)) ;
    }

    C->nvals = cnz ;
    C->magic = GB_MAGIC ;
    C->nvec_nonempty = M->nvec_nonempty ;
    C->jumbled = GB_JUMBLED (M) ;   // C is jumbled if M is jumbled

    GBURBLE ("(GPU C created and copied from M) ") ;

    //--------------------------------------------------------------------------
    // prefetch A and B
    //--------------------------------------------------------------------------

    // M might be very very sparse.  A(:,i) is not needed if M(:,i) is empty.
    // Likewise, B(:,j) is not needed if M(:,j) is empty.  For now, try this
    // heuristic:  if M is hypersparse, then do not prefetch A->b or A->x.

    int prefetch_b = (M_is_hyper) ? 0 : GB_PREFETCH_B ;
    int prefetch_x = (M_is_hyper) ? 0 : GB_PREFETCH_X ;
    int prefetch_pybi = GB_PREFETCH_PYI + prefetch_b ;

    // prefetch A (if A hypersparse: using A->Y)
    GB_OK (GB_cuda_matrix_prefetch (A, prefetch_pybi +
        (A_is_pattern ? 0 : prefetch_x), device, stream)) ;

    // prefetch B (if B hypersparse: using B->Y)
    GB_OK (GB_cuda_matrix_prefetch (B, prefetch_pybi +
        (B_is_pattern ? 0 : prefetch_x), device, stream)) ;

    //--------------------------------------------------------------------------
    // C<M>=A'*B on CUDA, in the JIT
    //--------------------------------------------------------------------------

//  final call looks like this:
//  GB_OK (GB_cuda_AxB_dot3_jit (C, M, Mask_struct, A, B, semiring, flipxy,
//      stream, device, number_of_sms)) ;

//  debugging for now, to die early if the CUDA fails to compile, load, or run:
    info = GB_cuda_AxB_dot3_jit (C, M, Mask_struct, A, B, semiring, flipxy,
        stream, device, number_of_sms) ;
    if (info == GrB_NO_VALUE) info = GrB_PANIC ;
    GB_OK (info) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    return GrB_SUCCESS;
}

