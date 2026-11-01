//------------------------------------------------------------------------------
// GraphBLAS/CUDA/select/GB_cuda_select_sparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "select/GB_cuda_select.hpp"

#undef  GB_FREE_ALL
#define GB_FREE_ALL                         \
{                                           \
    GB_phybix_free (C) ;                    \
    GB_cuda_stream_pool_release (&stream) ; \
}

GrB_Info GB_cuda_select_sparse
(
    GrB_Matrix C,               // C is jumbled if A is jumbled
    const bool C_iso,
    const GrB_IndexUnaryOp op,
    const bool flipij,
    const GrB_Matrix A,         // A can be jumbled, in all cases
    const GB_void *athunk,
    const GB_void *ythunk,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_NO_VALUE ;
    cudaStream_t stream = nullptr ;
    ASSERT (C != NULL) ;
    ASSERT (A != NULL) ;

    int data_arena = C->data_arena ;
    int device = data_arena - GxB_NARENAS ;

    GBURBLE ("(select sparse on cuda) ") ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    GB_OK (GB_cuda_stream_pool_acquire (device, &stream)) ;

    GB_OK (GB_wait_arenas (C)) ;
    GB_OK (GB_wait_arenas (A)) ;

    int64_t anz = GB_nnz_held (A) ;

    int32_t number_of_sms = GB_Global_gpu_sm_get (device) ;
    int64_t raw_gridsz = GB_ICEIL (anz, GB_CUDA_SELECT_SPARSE_CHUNKSIZE1) ;
    int32_t gridsz = std::min (raw_gridsz, (int64_t) (number_of_sms * 256)) ;
    gridsz = std::max (gridsz, 1) ;

    GBURBLE ("(cuda select sparse, device %d, sms: %d, gridsz: %d) ",
        device, number_of_sms, gridsz) ;

    //--------------------------------------------------------------------------
    // allocate the output matrix C
    //--------------------------------------------------------------------------

    // determine the p_is_32, j_is_32, and i_is_32 settings for the new matrix
    int csparsity = GxB_HYPERSPARSE ;
    bool Cp_is_32, Cj_is_32, Ci_is_32 ;
    GB_determine_pji_is_32 (&Cp_is_32, &Cj_is_32, &Ci_is_32,
        csparsity, anz, A->vlen, A->vdim, Werk) ;

    // Initialize C to be a user-returnable hypersparse empty matrix.
    // If needed, we handle the hyper->sparse conversion below.

    GB_OK (GB_new (&C, // sparse or hyper (from A), existing header
        A->type, A->vlen, A->vdim, GB_ph_calloc, A->is_csc,
        csparsity, A->hyper_switch, /* C->plen: revised later: */ 1,
        Cp_is_32, Cj_is_32, Ci_is_32, data_arena, data_arena)) ;

    C->iso = C_iso ;

    //--------------------------------------------------------------------------
    // C = select (A)
    //--------------------------------------------------------------------------

    GB_OK (GB_cuda_select_sparse_jit (C, A, flipij, ythunk, op,
        device, stream, gridsz)) ;

    //--------------------------------------------------------------------------
    // release the stream and finalize C
    //--------------------------------------------------------------------------

    GB_OK (GB_cuda_stream_pool_release (&stream)) ;

    ASSERT (C->x != NULL) ;

    if (C_iso)
    {
        // If C is iso, initialize the iso entry
        GB_select_iso ((GB_void *) C->x, op->opcode, athunk,
            (GB_void *) A->x, A->type->size) ;
    }

    if (C->nvec == C->vdim)
    {
        // C hypersparse with all vectors present; quick convert to sparse
        GB_FREE_MEMORY (&(C->h), C->h_mem) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "C output of cuda_select_sparse", GB0) ;
    return GrB_SUCCESS ;
}

