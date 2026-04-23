//------------------------------------------------------------------------------
// GraphBLAS/CUDA/builder/GB_cuda_builder.cpp
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "builder/GB_cuda_builder.hpp"

#undef  GB_FREE_ALL
#define GB_FREE_ALL                             \
{                                               \
    GB_Matrix_free (Thandle) ;                  \
    GB_cuda_stream_pool_release (&stream) ;     \
}

/* Alternative/additional signature:

check if I,J,X are not accessible by the GPU; if so, do phase1 on the CPU
    with OpenMP, here or in the caller.

--------------------------------------------------------------------------------
Usage in all of GraphBLAS:

(1) GB_build.c, for GrB_Matrix_build etc: DONE
    family: build.  I,J,X, no Key_in type, suffix: dup->name
    I,J,X are owned by the user.  Might not be accessible on the GPU.
    Must check for duplicates, need to sort.
    Must check if tuples are valid.

(2) GrB_Matrix_import:
    Same as GrB_Matrix_build.

(3) GB_concat_hyper:
    family: apply.  A, Key_in type, suffix: A->type (or second op->type)
    Its CUDA kernel must fill Key_in and input X from extractTuples.
    X is workspace to be freed so it can be reused by the builder
    No duplicates, need to sort; transplant of Sx is possible
    Tuples are known to be valid.

(4) GB_I_inverse:
    family: apply.  no A, need I, Key_in type, suffix: none
    J might be owned by the user.  Might not be accessible on the GPU.
    Its CUDA kernel must fill Key_in.  Matrix is iso.
    No duplicates, need to sort; transplant of Sx is possible
    Tuples are known to be valid.

(5) GB_hyper_hash_build:
    family: apply.  needs A, Key_in type, suffix: none
    Its CUDA kernel must fill Key_in and X.
    X is workspace to be freed so it can be reused by the builder
    No duplicates, need to sort; transplant of Sx is possible
    Tuples are known to be valid.

(6) GB_reshape:
    family: apply.  needs A, Key_in type, suffix: A->type
    Its CUDA kernel must fill Key_in.  Matrix can be iso or non-iso.
    might be in-place (X is consumed here) or not in-place (X is readonly)
    X is workspace to be freed so it can be reused by the builder
    No duplicates, might need to sort if input matrix is jumbled;
    if so then transplant of Sx is possible.  Otherwise, 
    GB_cuda_builder_phase5_no_dupl will copy Sx (== A->x) into T->x.
    Tuples are known to be valid.

(7) GB_transpose_builder: DONE
    family: apply.  needs A, Key_in type, suffix: A->type
    Its CUDA kernel must fill Key_in.  Matrix can be iso or non-iso.
    X is the input A->x and cannot be modified ... unless A is transposed
        in place.  However, the sort is out-of-place so re-using A->x
        does not help.  Instead, since Sx is workspace (output from CUB
        sort) and no duplicates appear, Sx can be transplanted as T->x if
        no typecasting is needed
    No duplicates, need to sort.
    Tuples are known to be valid.

(8) GB_wait:
    family: ewise.  no A, needs A->Pending (as I,J,X), Key_in type,
        suffix: A->pending->op->name (which is dup->name)
    Its CUDA kernel must fill Key_in.  Matrix can be iso or non-iso.
    Must check for duplicates, need to sort (depending on A->pending->sorted)
    Tuples are known to be valid.

    If no typecasting and A->type and A->Pending->type match, then
    A+build(A->Pending) could skip ewiseAdd, and instead place all of A in the
    keys, and do build (A, A->Pending).

General observations:

    (1) if Key_in is provided, no need to check if tuples are valid
    (2) input Key_in might be known to be sorted.  If so, then
        GB_KNOWN_SORTED is set true in the JIT kernel.
    (3) duplicates can appear in Key_in (GB_wait only); other cases have
        no duplicates.  If duplicates cannot possibly appear then
        GB_KNOWN_NO_DUPLICATES is set true in the JIT kernel.

*/

GrB_Info GB_cuda_builder            // build a matrix from tuples
(
    // output, not defined on input:
    GrB_Matrix *Thandle,    // matrix to build, dynamic header
    // inputs, not modified:
    const GrB_Type ttype,   // type of output matrix T
    const int64_t vlen,     // length of each vector of T
    const int64_t vdim,     // number of vectors in T
    const bool is_csc,      // true if T is CSC, false if CSR
    const bool is_matrix,   // true if T a GrB_Matrix, false if vector
    const GB_void *Key_input,  // if Key is preloaded, NULL otherwise
    const GB_void *I,       // original indices, size nvals
    const GB_void *J,       // original indices, size nvals
    const GB_void *X,       // array of values of tuples, size nvals,
                            // or size 1 if X is iso
    const bool X_iso,       // true if X is iso
    const int64_t nvals,    // number of tuples
    GrB_BinaryOp dup,       // binary function to assemble duplicates,
                            // if NULL use the SECOND operator to
                            // keep the most recent duplicate.
    const GrB_Type xtype,   // the type of X
    bool do_burble,         // if true, then burble is allowed
    bool I_is_32,       // true if I is 32 bit, false if 64
    bool J_is_32,       // true if J is 32 bit, false if 64
    bool Tp_is_32,      // true if T->p is built as 32 bit, false if 64
    bool Tj_is_32,      // true if T->h is built as 32 bit, false if 64
    bool Ti_is_32,      // true if T->i is built as 32 bit, false if 64
    bool known_no_duplicates,   // true if tuples known to have no duplicates
    bool known_sorted           // true if tuples known to be sorted on input
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_NO_VALUE ;
    ASSERT (Thandle != NULL) ;
    ASSERT (I != NULL || Key_input != NULL) ;
    ASSERT (X != NULL) ;
    ASSERT (ttype != NULL) ;
    ASSERT (xtype != NULL) ;

    //--------------------------------------------------------------------------
    // construct the SECOND operator if dup is NULL
    //--------------------------------------------------------------------------

    struct GB_BinaryOp_opaque dup_header ;
    if (dup == NULL)
    { 
        // z = SECOND (x,y) where all three types are the same as ttype
        // T(i,j) = (ttype) Sx(k) will be done for all tuples.  If dup is
        // SECOND_UDT, dup->binop_function will be NULL; this is OK.
        dup = GB_binop_second (ttype, &dup_header) ;
        ASSERT (dup != NULL && GB_op_is_second (dup, ttype)) ;
    }

    //--------------------------------------------------------------------------
    // get CUDA stream and geometry
    //--------------------------------------------------------------------------

    cudaStream_t stream = nullptr ;
    GB_OK (GB_cuda_stream_pool_acquire (&stream)) ;

    // determine the geometry of the CUDA kernel launches
    int32_t number_of_sms = GB_Global_gpu_sm_get (0) ;
    int64_t raw_gridsz = GB_ICEIL (nvals, GB_CUDA_BUILDER_CHUNKSIZE) ;
    int32_t gridsz = std::min (raw_gridsz, (int64_t) (number_of_sms * 256)) ;
    gridsz = std::max (gridsz, 1) ;

    //--------------------------------------------------------------------------
    // build T from the (I,J,X) tuples
    //--------------------------------------------------------------------------

    GB_OK (GB_cuda_builder_jit (Thandle, ttype, vlen, vdim, is_csc, is_matrix,
        Key_input, I, J, X, X_iso, nvals, dup, xtype, I_is_32, J_is_32,
        Tp_is_32, Tj_is_32, Ti_is_32, known_no_duplicates, known_sorted,
        stream, gridsz)) ;

    //--------------------------------------------------------------------------
    // release the stream
    //--------------------------------------------------------------------------

    GB_OK (GB_cuda_stream_pool_release (&stream)) ;

    //--------------------------------------------------------------------------
    // handle the iso case
    //--------------------------------------------------------------------------

    if (X_iso)
    {
        // copy the iso-value entry of X [0] into T->x [0]
        ASSERT (xtype == ttype) ;
        GrB_Matrix T = (*Thandle) ;
        memcpy (T->x, X, xtype->size) ;
    }

    // return the result
    return (GrB_SUCCESS) ;
}

