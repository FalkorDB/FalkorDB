//------------------------------------------------------------------------------
// GB_meta16_definitions.h: methods that depend on the sparsity of A and B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Define macros that depend on the sparsity of A and B for GB_meta16_factory.
// These macros are only used by saxpy3 methods.

//------------------------------------------------------------------------------
// GB_GET_B_j: prepare to iterate over B(:,j)
//------------------------------------------------------------------------------

#undef GB_GET_B_j

#if defined ( GB_META16 )

    #if ( GB_B_IS_HYPER || GB_A_IS_HYPER )

        // A or B are hyper
        #define GB_GET_B_j \
        GB_GET_B_j_FOR_ALL_FORMATS (GB_A_IS_HYPER,GB_B_IS_SPARSE,GB_B_IS_HYPER)

    #else

        #if ( GB_B_IS_SPARSE )

            // B is sparse
            #define GB_GET_B_j                              \
                const int64_t j = kk ;                      \
                int64_t pB = Bp [kk] ;                      \
                const int64_t pB_end = Bp [kk+1] ;          \
                const int64_t bjnz = pB_end - pB ;          \
                GB_GET_T_FOR_SECONDJ

        #else

            // B is bitmap or full
            #define GB_GET_B_j                              \
                const int64_t j = kk ;                      \
                int64_t pB = kk * bvlen ;                   \
                const int64_t pB_end = pB + bvlen ;         \
                const int64_t bjnz = bvlen ;                \
                GB_GET_T_FOR_SECONDJ

        #endif

    #endif

#else

    // define GB_GET_B_j for all sparsity formats
    #define GB_GET_B_j \
        GB_GET_B_j_FOR_ALL_FORMATS (A_is_hyper, B_is_sparse, B_is_hyper)

#endif

//------------------------------------------------------------------------------
// GB_GET_B_kj_INDEX: get the index k of the entry B(k,j)
//------------------------------------------------------------------------------

#undef GB_GET_B_kj_INDEX

#if defined ( GB_META16 )

    #if ( GB_B_IS_HYPER || GB_B_IS_SPARSE )

        // B is hyper or sparse
        #define GB_GET_B_kj_INDEX                   \
            const int64_t k = Bi [pB]

    #elif ( GB_B_IS_BITMAP )

        // B is bitmap
        #define GB_GET_B_kj_INDEX                   \
            if (!Bb [pB]) continue ;                \
            const int64_t k = pB % bvlen

    #else

        // B is full
        #define GB_GET_B_kj_INDEX                   \
            const int64_t k = pB % bvlen

    #endif

#else

    // for any format of B
    #define GB_GET_B_kj_INDEX                       \
        if (!GBB_B (Bb, pB)) continue ;             \
        const int64_t k = GBI_B (Bi, pB, bvlen)

#endif

//------------------------------------------------------------------------------
// GB_GET_A_k: prepare to iterate over the vector A(:,k)
//------------------------------------------------------------------------------

#undef GB_GET_A_k

#if defined ( GB_META16 )

    #if ( GB_A_IS_HYPER )

        // A is hyper
        #define GB_GET_A_k GB_GET_A_k_FOR_ALL_FORMATS (true)

    #elif ( GB_A_IS_SPARSE )

        // A is sparse
        #define GB_GET_A_k                              \
            const int64_t pA_start = Ap [k] ;           \
            const int64_t pA_end = Ap [k+1] ;           \
            const int64_t aknz = pA_end - pA_start

    #else

        // A is bitmap or full
        #define GB_GET_A_k                              \
            const int64_t pA_start = k * avlen ;        \
            const int64_t pA_end = pA_start + avlen ;   \
            const int64_t aknz = avlen

    #endif

#else

    // define GB_GET_A_k for all sparsity formats
    #define GB_GET_A_k GB_GET_A_k_FOR_ALL_FORMATS (A_is_hyper)

#endif

//------------------------------------------------------------------------------
// GB_GET_A_ik_INDEX: get the index i of the entry A(i,k)
//------------------------------------------------------------------------------

#undef GB_GET_A_ik_INDEX

#if defined ( GB_META16 )

    #if ( GB_A_IS_HYPER || GB_A_IS_SPARSE )

        // A is hyper or sparse
        #define GB_GET_A_ik_INDEX                   \
            const int64_t i = Ai [pA]

    #elif ( GB_A_IS_BITMAP )

        // A is bitmap
        #define GB_GET_A_ik_INDEX                   \
            if (!Ab [pA]) continue ;                \
            const int64_t i = pA % avlen

    #else

        // A is full
        #define GB_GET_A_ik_INDEX                   \
            const int64_t i = pA % avlen

    #endif

#else

    // for any format of A
    #define GB_GET_A_ik_INDEX                       \
        if (!GBB_A (Ab, pA)) continue ;             \
        const int64_t i = GBI_A (Ai, pA, avlen)

#endif

//------------------------------------------------------------------------------
// GB_COMPUTE_C_j_WHEN_NNZ_B_j_IS_ONE: compute C(:,j) when nnz(B(:,j)) == 1
//------------------------------------------------------------------------------

// C(:,j) = A(:,k)*B(k,j) when there is a single entry in B(:,j)
// The mask must not be present.  A must be sparse or hypersparse.

#undef GB_COMPUTE_C_j_WHEN_NNZ_B_j_IS_ONE

#if GB_IS_ANY_PAIR_SEMIRING

    // ANY_PAIR: result is purely symbolic; no numeric work to do
    #define GB_COMPUTE_C_j_WHEN_NNZ_B_j_IS_ONE                      \
        ASSERT (A_is_sparse || A_is_hyper) ;                        \
        GB_GET_B_kj_INDEX ;         /* get index k of B(k,j) */     \
        GB_GET_A_k ;                /* get A(:,k) */                \
        memcpy (Ci + pC, Ai + pA_start, aknz * sizeof (int64_t)) ;  \
        /* C becomes jumbled if A is jumbled */                     \
        task_C_jumbled = task_C_jumbled || A_jumbled ;

#else

    // typical semiring
    #define GB_COMPUTE_C_j_WHEN_NNZ_B_j_IS_ONE                      \
        ASSERT (A_is_sparse || A_is_hyper) ;                        \
        GB_GET_B_kj_INDEX ;         /* get index k of B(k,j) */     \
        GB_GET_A_k ;                /* get A(:,k) */                \
        GB_GET_B_kj ;               /* bkj = B(k,j) */              \
        /* scan A(:,k) */                                           \
        for (int64_t pA = pA_start ; pA < pA_end ; pA++)            \
        {                                                           \
            GB_GET_A_ik_INDEX ;     /* get index i of A(i,k) */     \
            GB_MULT_A_ik_B_kj ;         /* t = A(i,k)*B(k,j) */     \
            GB_CIJ_WRITE (pC, t) ;      /* Cx [pC] = t */           \
            Ci [pC++] = i ;                                         \
        }                                                           \
        /* C becomes jumbled if A is jumbled */                     \
        task_C_jumbled = task_C_jumbled || A_jumbled ;

#endif

//------------------------------------------------------------------------------
// GB_COMPUTE_DENSE_C_j: compute C(:,j)=A*B(:,j) when C(:,j) is completely dense
//------------------------------------------------------------------------------

// This method is not used for the saxpy3 generic method.
// It is only used by the GB_AxB_saxpy3_coarseGus_*_phase5.c methods
// for pre-generated, any_pair (see the case below), and JIT kernels.

#undef GB_COMPUTE_DENSE_C_j

#if GB_IS_ANY_PAIR_SEMIRING

    // ANY_PAIR: result is purely symbolic; no numeric work to do
    #define GB_COMPUTE_DENSE_C_j                                    \
        for (int64_t i = 0 ; i < cvlen ; i++)                       \
        {                                                           \
            Ci [pC + i] = i ;                                       \
        }

#else

    // typical semiring
    #define GB_COMPUTE_DENSE_C_j                                    \
        for (int64_t i = 0 ; i < cvlen ; i++)                       \
        {                                                           \
            Ci [pC + i] = i ;                                       \
            GB_CIJ_WRITE (pC + i, zidentity) ;  /* C(i,j)=0 */      \
        }                                                           \
        for ( ; pB < pB_end ; pB++)     /* scan B(:,j) */           \
        {                                                           \
            GB_GET_B_kj_INDEX ;         /* get index k of B(k,j) */ \
            GB_GET_A_k ;                /* get A(:,k) */            \
            if (aknz == 0) continue ;   /* skip if A(:,k) empty */  \
            GB_GET_B_kj ;               /* bkj = B(k,j) */          \
            /* scan A(:,k) */                                       \
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)        \
            {                                                       \
                GB_GET_A_ik_INDEX ;     /* get index i of A(i,k) */ \
                GB_MULT_A_ik_B_kj ;     /* t = A(i,k)*B(k,j) */     \
                GB_CIJ_UPDATE (pC + i, t) ; /* Cx [pC+i]+=t */      \
            }                                                       \
        }

#endif

//------------------------------------------------------------------------------
// GB_SCAN_M_j_OR_A_k: compute C(:,j) using linear scan or binary search
//------------------------------------------------------------------------------

// C(:,j)<M(:,j)>=A(:,k)*B(k,j) using one of two methods
#undef  GB_SCAN_M_j_OR_A_k

#define GB_SCAN_M_j_OR_A_k(A_ok_for_binary_search)                          \
{                                                                           \
    if (A_ok_for_binary_search && aknz > 256 && mjnz_much < aknz &&         \
        mjnz < mvlen && aknz < avlen)                                       \
    {                                                                       \
        /* M and A are both sparse, and nnz(M(:,j)) is much less than */    \
        /* nnz(A(:,k)); scan M(:,j), and do binary search for A(i,k).*/     \
        /* This requires that A is not jumbled. */                          \
        int64_t pA = pA_start ;                                             \
        for (int64_t pM = pM_start ; pM < pM_end ; pM++)                    \
        {                                                                   \
            GB_GET_M_ij (pM) ;      /* get M(i,j) */                        \
            if (!mij) continue ;    /* skip if M(i,j)=0 */                  \
            int64_t i = Mi [pM] ;                                           \
            bool found ;            /* search for A(i,k) */                 \
            if (M_jumbled) pA = pA_start ;                                  \
            int64_t apright = pA_end - 1 ;                                  \
            GB_BINARY_SEARCH (i, Ai, pA, apright, found) ;                  \
            if (found)                                                      \
            {                                                               \
                /* C(i,j)<M(i,j)> += A(i,k) * B(k,j) for this method. */    \
                /* M(i,j) is always 1, as given in the hash table */        \
                GB_IKJ ;                                                    \
            }                                                               \
        }                                                                   \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        /* A(:,j) is sparse enough relative to M(:,j) */                    \
        /* M and/or A can dense, and either can be jumbled. */              \
        /* scan A(:,k), and lookup M(i,j) (in the hash table) */            \
        for (int64_t pA = pA_start ; pA < pA_end ; pA++)                    \
        {                                                                   \
            GB_GET_A_ik_INDEX ;     /* get index i of A(i,j) */             \
            /* do C(i,j)<M(i,j)> += A(i,k) * B(k,j) for this method */      \
            /* M(i,j) may be 0 or 1, as given in the hash table */          \
            GB_IKJ ;                                                        \
        }                                                                   \
    }                                                                       \
}

