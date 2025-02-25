
#include "GB_cuda_select.hpp"

#undef  GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE                                   \
{                                                           \
    if (stream != nullptr)                                  \
    {                                                       \
        cudaStreamSynchronize (stream) ;                    \
        cudaStreamDestroy (stream) ;                        \
    }                                                       \
    stream = nullptr ;                                      \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL         \
{                           \
    GB_phybix_free (C) ;    \
    GB_FREE_WORKSPACE ;     \
}

#define BLOCK_SIZE 512
#define LOG2_BLOCK_SIZE 9

GrB_Info GB_cuda_select_sparse
(
    GrB_Matrix C,
    const bool C_iso,
    const GrB_IndexUnaryOp op,
    const bool flipij,
    const GrB_Matrix A,
    const GB_void *athunk,
    const GB_void *ythunk,
    GB_Werk Werk
)
{

    // check inputs
    GrB_Info info = GrB_NO_VALUE ;
    ASSERT (C != NULL && !(C->header_size == 0)) ;
    ASSERT (A != NULL && !(A->header_size == 0)) ;

    // FIXME: use the stream pool
    cudaStream_t stream = nullptr ;
    CUDA_OK (cudaStreamCreate (&stream)) ;

    GrB_Index anz = GB_nnz_held (A) ;

    int32_t number_of_sms = GB_Global_gpu_sm_get (0) ;
    int64_t raw_gridsz = GB_ICEIL (anz, BLOCK_SIZE) ;
    int32_t gridsz = std::min (raw_gridsz, (int64_t) (number_of_sms * 256)) ;
    gridsz = std::max (gridsz, 1) ;

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
        Cp_is_32, Cj_is_32, Ci_is_32)) ;

    C->jumbled = A->jumbled ;
    C->iso = C_iso ;

    GB_OK (GB_cuda_select_sparse_jit (C, A,
        flipij, ythunk, op, stream, gridsz, BLOCK_SIZE)) ;

    GB_FREE_WORKSPACE ;

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
        GB_FREE_MEMORY (&(C->h), C->h_size) ;
    }

    ASSERT_MATRIX_OK (C, "C output of cuda_select_sparse", GB0) ;
    return GrB_SUCCESS ;
}

