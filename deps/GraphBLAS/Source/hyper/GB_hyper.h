//------------------------------------------------------------------------------
// GB_hyper.h: definitions for hypersparse matrices and related methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_HYPER_H
#define GB_HYPER_H

int64_t GB_nvec_nonempty        // return # of non-empty vectors
(
    const GrB_Matrix A          // input matrix to examine
) ;

static inline int64_t GB_nvec_nonempty_get
(
    GrB_Matrix A
)
{
    int64_t nvec_nonempty = 0 ;
    if (A != NULL)
    { 
        GB_ATOMIC_READ
        nvec_nonempty = A->nvec_nonempty ;
    }
    return (nvec_nonempty) ;
}

static inline void GB_nvec_nonempty_set
(
    GrB_Matrix A,
    int64_t nvec_nonempty
)
{
    if (A != NULL)
    { 
        GB_ATOMIC_WRITE
        A->nvec_nonempty = nvec_nonempty ;
    }
}

static inline int64_t GB_nvec_nonempty_update
(
    GrB_Matrix A
)
{
    int64_t nvec_nonempty = 0 ;
    if (A != NULL)
    {
        // get the current value of A->nvec_nonempty
        nvec_nonempty = GB_nvec_nonempty_get (A) ;
        if (nvec_nonempty < 0)
        { 
            // compute A->nvec_nonempty and then update it atomically
            nvec_nonempty = GB_nvec_nonempty (A) ;
            GB_nvec_nonempty_set (A, nvec_nonempty) ;
        }
    }
    return (nvec_nonempty) ;
}

GrB_Info GB_hyper_realloc
(
    GrB_Matrix A,               // matrix with hyperlist to reallocate
    int64_t plen_new,           // new size of A->p and A->h
    GB_Werk Werk
) ;

GrB_Info GB_hyper_prune
(
    GrB_Matrix A,               // matrix to prune
    GB_Werk Werk
) ;

bool GB_hyper_hash_need         // return true if A needs a hyper hash
(
    GrB_Matrix A
) ;

GrB_Matrix GB_hyper_shallow         // return C
(
    GrB_Matrix C,                   // output matrix
    const GrB_Matrix A              // input matrix
) ;

#endif

