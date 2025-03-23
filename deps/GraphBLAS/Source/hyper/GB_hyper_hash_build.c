//------------------------------------------------------------------------------
// GB_hyper_hash_build: construct A->Y for a hypersparse matrix A
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_FREE_WORKSPACE               \
{                                       \
    GB_FREE_MEMORY (&I_work, I_work_size) ;    \
    GB_FREE_MEMORY (&J_work, J_work_size) ;    \
    GB_FREE_MEMORY (&X_work, X_work_size) ;    \
}

#define GB_FREE_ALL                     \
{                                       \
    GB_FREE_WORKSPACE ;                 \
    GB_phybix_free (A) ;                \
}

#include "builder/GB_build.h"

GB_CALLBACK_HYPER_HASH_BUILD_PROTO (GB_hyper_hash_build)
{
    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (!GB_hyper_hash_need (A))
    { 
        // quick return: A is NULL, not hypersparse, A->Y already computed,
        // or A does not have enough non-empty vectors to warrant the creation
        // of the A->Y hyper_hash
        return (GrB_SUCCESS) ;
    }

    GrB_Info info ;
    GB_MDECL (I_work, , u) ; size_t I_work_size = 0 ;
    GB_MDECL (J_work, , u) ; size_t J_work_size = 0 ;
    GB_MDECL (X_work, , u) ; size_t X_work_size = 0 ;

    ASSERT_MATRIX_OK (A, "A for hyper_hash", GB0) ;
    GB_BURBLE_MATRIX (A, "(build hyper hash) ") ;

    //--------------------------------------------------------------------------
    // allocate A->Y
    //--------------------------------------------------------------------------

    // A->Y is (A->vdim)-by-(hash table size for A->h), with one vector per
    // hash bucket.

    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
    ASSERT (Ah != NULL) ;

    int64_t anvec = A->nvec ;
    // this ensures a load factor of 0.5 to 1:
    int64_t yvdim = ((uint64_t) 1) << (GB_FLOOR_LOG2 (anvec) + 1) ;
    // divide by 4 to get a load factor of 2 to 4:
    yvdim = yvdim / 4 ;
    yvdim = GB_IMAX (yvdim, 4) ;
    int64_t yvlen = A->vdim ;
    int64_t hash_bits = (yvdim - 1) ;   // yvdim is always a power of 2
    bool Aj_is_32 = A->j_is_32 ;

    GB_OK (GB_new (&(A->Y), // new dynamic header, do not allocate any content
        GrB_UINT64, yvlen, yvdim, GB_ph_null, true, GxB_SPARSE, -1, 0,
        Aj_is_32, Aj_is_32, Aj_is_32)) ;
    GrB_Matrix Y = A->Y ;

    //--------------------------------------------------------------------------
    // create the tuples for A->Y
    //--------------------------------------------------------------------------

    size_t jsize = (Aj_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
    I_work = GB_MALLOC_MEMORY (anvec, jsize, &I_work_size) ;
    J_work = GB_MALLOC_MEMORY (anvec, jsize, &J_work_size) ;
    X_work = GB_MALLOC_MEMORY (anvec, jsize, &X_work_size) ;
    if (I_work == NULL || J_work == NULL || X_work == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    GB_IPTR (I_work, Aj_is_32) ;
    GB_IPTR (J_work, Aj_is_32) ;
    GB_IPTR (X_work, Aj_is_32) ;

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (anvec, chunk, nthreads_max) ;

    int64_t k ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (k = 0 ; k < anvec ; k++)
    {
        uint64_t j = GB_IGET (Ah, k) ;
        uint64_t hash = GB_HASHF2 (j, hash_bits) ;     // in range 0 to yvdim-1
        // J_work [k] = hash ;
        GB_ISET (J_work, k, hash) ;
        // I_work [k] = j ;
        GB_ISET (I_work, k, j) ;
        // X_work [k] = k ;
        GB_ISET (X_work, k, k) ;
    }

    //--------------------------------------------------------------------------
    // build A->Y, initially hypersparse
    //--------------------------------------------------------------------------

    GrB_Type ytype = (Aj_is_32) ? GrB_UINT32 : GrB_UINT64 ;

    GB_OK (GB_builder (
        Y,                      // create Y using a dynamic header
        ytype,                  // Y->type
        yvlen,                  // Y->vlen
        yvdim,                  // Y->vdim
        true,                   // Y->is_csc
        &I_work,                // row indices
        &I_work_size,
        &J_work,                // column indices
        &J_work_size,
        (GB_void **) &X_work,   // values
        &X_work_size,
        false,                  // tuples need to be sorted
        true,                   // no duplicates
        anvec,                  // size of I_work and J_work in # of tuples
        true,                   // is_matrix: unused
        NULL, NULL,             // original I,J tuples
        NULL,                   // no scalar iso value
        false,                  // Y is never iso
        anvec,                  // # of tuples
        NULL,                   // no duplicates, so dup is NUL
        ytype,                  // the type of X_work
        false,                  // no burble (already burbled above)
        Werk,
        Aj_is_32, Aj_is_32,     // if true, [IJ]_work 32-bit, else 64-bit
        Aj_is_32, Aj_is_32, Aj_is_32  // integer size of A->Y->[pix]
    )) ;

    Y->hyper_switch = -1 ;              // never make Y hypersparse
    Y->sparsity_control = GxB_SPARSE ;  // Y is always sparse CSC
    ASSERT (GB_IS_HYPERSPARSE (Y)) ;    // Y is currently hypersparse

    // workspace has been freed by GB_builder, or transplanted in to Y
    ASSERT (I_work == NULL) ;
    ASSERT (J_work == NULL) ;
    ASSERT (X_work == NULL) ;

    //--------------------------------------------------------------------------
    // convert A->Y to sparse
    //--------------------------------------------------------------------------

    // GB_builder always constructs its matrix as hypersparse.  Y is now
    // conformed to its required sparsity format: always sparse.  No burble;
    // (already burbled above).

    GB_OK (GB_convert_hyper_to_sparse (Y, false)) ;
    ASSERT (anvec == GB_nnz (Y)) ;
    ASSERT (GB_IS_SPARSE (Y)) ;         // Y is now sparse and will remain so

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A from hyper_hash", GB0) ;
    ASSERT (!GB_ZOMBIES (Y)) ;
    ASSERT (!GB_JUMBLED (Y)) ;
    ASSERT (!GB_PENDING (Y)) ;
    return (GrB_SUCCESS) ;
}

