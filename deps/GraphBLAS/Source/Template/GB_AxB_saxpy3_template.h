//------------------------------------------------------------------------------
// GB_AxB_saxpy3_template.h: C=A*B, C<M>=A*B, or C<!M>=A*B via saxpy3 method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Definitions for GB_AxB_saxpy3_template.c.  These do not depend on the
// sparsity of A and B.

#ifndef GB_AXB_SAXPY3_TEMPLATE_H
#define GB_AXB_SAXPY3_TEMPLATE_H

//------------------------------------------------------------------------------
// GB_GET_M_j: prepare to iterate over M(:,j)
//------------------------------------------------------------------------------

// prepare to iterate over the vector M(:,j), for the (kk)th vector of B

#define GB_GET_M_j                                                          \
    int64_t pM_start, pM_end ;                                              \
    if (M_is_hyper)                                                         \
    {                                                                       \
        /* M is hypersparse: find M(:,j) in the M->Y hyper_hash */          \
        GB_hyper_hash_lookup (Mh, mnvec, Mp, M_Yp, M_Yi, M_Yx, M_hash_bits, \
            GBH_B (Bh, kk), &pM_start, &pM_end) ;                           \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        /* A is sparse, bitmap, or full */                                  \
        int64_t j = GBH_B (Bh, kk) ;                                        \
        pM_start = GBP_M (Mp, j  , mvlen) ;                                 \
        pM_end   = GBP_M (Mp, j+1, mvlen) ;                                 \
    }                                                                       \
    const int64_t mjnz = pM_end - pM_start

//------------------------------------------------------------------------------
// GB_GET_M_j_RANGE
//------------------------------------------------------------------------------

#define GB_GET_M_j_RANGE(gamma) \
    const int64_t mjnz_much = mjnz * gamma

//------------------------------------------------------------------------------
// GB_SCATTER_Mj_t: scatter M(:,j) of the given type into Gus. workspace
//------------------------------------------------------------------------------

#ifndef GB_JIT_KERNEL

    // for generic pre-generated kernels only; not needed for JIT kernels
    #define GB_SCATTER_Mj_t(mask_t,pMstart,pMend,mark)                      \
    {                                                                       \
        const mask_t *restrict Mxx = (mask_t *) Mx ;                        \
        if (M_is_bitmap)                                                    \
        {                                                                   \
            /* M is bitmap */                                               \
            for (int64_t pM = pMstart ; pM < pMend ; pM++)                  \
            {                                                               \
                /* if (M (i,j) == 1) mark Hf [i] */                         \
                if (Mb [pM] && Mxx [pM]) Hf [GBI_M (Mi, pM, mvlen)] = mark ;\
            }                                                               \
        }                                                                   \
        else                                                                \
        {                                                                   \
            /* M is hyper, sparse, or full */                               \
            for (int64_t pM = pMstart ; pM < pMend ; pM++)                  \
            {                                                               \
                /* if (M (i,j) == 1) mark Hf [i] */                         \
                if (Mxx [pM]) Hf [GBI_M (Mi, pM, mvlen)] = mark ;           \
            }                                                               \
        }                                                                   \
    }                                                                       \
    break ;

#endif

//------------------------------------------------------------------------------
// GB_SCATTER_M_j:  scatter M(:,j) into the Gustavson workpace
//------------------------------------------------------------------------------

#ifdef GB_JIT_KERNEL

    // for JIT kernels
    #define GB_SCATTER_M_j(pMstart,pMend,mark)                              \
        for (int64_t pM = pMstart ; pM < pMend ; pM++)                      \
        {                                                                   \
            /* if (M (i,j) is present) mark Hf [i] */                       \
            if (GBB_M (Mb,pM) && GB_MCAST (Mx,pM,))                         \
            {                                                               \
                Hf [GBI_M (Mi, pM, mvlen)] = mark ;                         \
            }                                                               \
        }

#else

    // for generic pre-generated kernels
    #define GB_SCATTER_M_j(pMstart,pMend,mark)                              \
        if (Mx == NULL)                                                     \
        {                                                                   \
            /* M is structural, not valued */                               \
            if (M_is_bitmap)                                                \
            {                                                               \
                /* M is bitmap */                                           \
                for (int64_t pM = pMstart ; pM < pMend ; pM++)              \
                {                                                           \
                    /* if (M (i,j) is present) mark Hf [i] */               \
                    if (Mb [pM]) Hf [GBI_M (Mi, pM, mvlen)] = mark ;        \
                }                                                           \
            }                                                               \
            else                                                            \
            {                                                               \
                /* M is hyper, sparse, or full */                           \
                for (int64_t pM = pMstart ; pM < pMend ; pM++)              \
                {                                                           \
                    /* mark Hf [i] */                                       \
                    Hf [GBI_M (Mi, pM, mvlen)] = mark ;                     \
                }                                                           \
            }                                                               \
        }                                                                   \
        else                                                                \
        {                                                                   \
            /* mask is valued, not structural */                            \
            switch (msize)                                                  \
            {                                                               \
                default:                                                    \
                case GB_1BYTE:                                              \
                    GB_SCATTER_Mj_t (uint8_t , pMstart, pMend, mark) ;      \
                case GB_2BYTE:                                              \
                    GB_SCATTER_Mj_t (uint16_t, pMstart, pMend, mark) ;      \
                case GB_4BYTE:                                              \
                    GB_SCATTER_Mj_t (uint32_t, pMstart, pMend, mark) ;      \
                case GB_8BYTE:                                              \
                    GB_SCATTER_Mj_t (uint64_t, pMstart, pMend, mark) ;      \
                case GB_16BYTE:                                             \
                {                                                           \
                    const uint64_t *restrict Mxx = (uint64_t *) Mx ;        \
                    for (int64_t pM = pMstart ; pM < pMend ; pM++)          \
                    {                                                       \
                        /* if (M (i,j) == 1) mark Hf [i] */                 \
                        if (!GBB_M (Mb, pM)) continue ;                     \
                        if (Mxx [2*pM] || Mxx [2*pM+1])                     \
                        {                                                   \
                            /* Hf [i] = M(i,j) */                           \
                            Hf [GBI_M (Mi, pM, mvlen)] = mark ;             \
                        }                                                   \
                    }                                                       \
                }                                                           \
            }                                                               \
        }

#endif

//------------------------------------------------------------------------------
// GB_HASH_M_j: scatter M(:,j) for a coarse hash task
//------------------------------------------------------------------------------

// hash M(:,j) into Hf and Hi for coarse hash task, C<M>=A*B or C<!M>=A*B
#define GB_HASH_M_j                                         \
    for (int64_t pM = pM_start ; pM < pM_end ; pM++)        \
    {                                                       \
        GB_GET_M_ij (pM) ;      /* get M(i,j) */            \
        if (!mij) continue ;    /* skip if M(i,j)=0 */      \
        const int64_t i = GBI_M (Mi, pM, mvlen) ;           \
        for (GB_HASH (i))       /* find i in hash */        \
        {                                                   \
            if (Hf [hash] < mark)                           \
            {                                               \
                Hf [hash] = mark ;  /* insert M(i,j)=1 */   \
                Hi [hash] = i ;                             \
                break ;                                     \
            }                                               \
        }                                                   \
    }

//------------------------------------------------------------------------------
// GB_GET_T_FOR_SECONDJ: define t for SECONDJ and SECONDJ1 semirings
//------------------------------------------------------------------------------

#if GB_IS_SECONDJ_MULTIPLIER
    #define GB_GET_T_FOR_SECONDJ                            \
        GB_CIJ_DECLARE (t) ;                                \
        GB_MULT (t, ignore, ignore, ignore, ignore, j) ;
#else
    #define GB_GET_T_FOR_SECONDJ
#endif

//------------------------------------------------------------------------------
// GB_GET_B_j_FOR_ALL_FORMATS: prepare to iterate over B(:,j)
//------------------------------------------------------------------------------

// prepare to iterate over the vector B(:,j), the (kk)th vector in B, where 
// j == GBH_B (Bh, kk).  This macro works regardless of the sparsity of A and B.
#define GB_GET_B_j_FOR_ALL_FORMATS(A_is_hyper,B_is_sparse,B_is_hyper)       \
    const int64_t j = (B_is_hyper) ? Bh [kk] : kk ;                         \
    GB_GET_T_FOR_SECONDJ ;  /* t = j for SECONDJ, or j+1 for SECONDJ1 */    \
    int64_t pB = (B_is_sparse || B_is_hyper) ? Bp [kk] : (kk * bvlen) ;     \
    const int64_t pB_end =                                                  \
        (B_is_sparse || B_is_hyper) ? Bp [kk+1] : (pB+bvlen) ;              \
    const int64_t bjnz = pB_end - pB ;  /* nnz (B (:,j) */

//------------------------------------------------------------------------------
// GB_GET_B_kj: get the numeric value of B(k,j)
//------------------------------------------------------------------------------

#if GB_IS_FIRSTJ_MULTIPLIER

    // FIRSTJ or FIRSTJ1 multiplier
    // t = aik * bkj = k or k+1
    #define GB_GET_B_kj                                     \
        GB_CIJ_DECLARE (t) ;                                \
        GB_MULT (t, ignore, ignore, ignore, k, ignore)

#else

    #define GB_GET_B_kj \
        GB_DECLAREB (bkj) ;                                 \
        GB_GETB (bkj, Bx, pB, B_iso)       /* bkj = Bx [pB] */

#endif

//------------------------------------------------------------------------------
// GB_GET_A_k_FOR_ALL_FORMATS: prepare to iterate over the vector A(:,k)
//------------------------------------------------------------------------------

#define GB_GET_A_k_FOR_ALL_FORMATS(A_is_hyper)                              \
    int64_t pA_start, pA_end ;                                              \
    if (A_is_hyper)                                                         \
    {                                                                       \
        /* A is hypersparse: find A(:,k) in the A->Y hyper_hash */          \
        GB_hyper_hash_lookup (Ah, anvec, Ap, A_Yp, A_Yi, A_Yx, A_hash_bits, \
            k, &pA_start, &pA_end) ;                                        \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        /* A is sparse, bitmap, or full */                                  \
        pA_start = GBP_A (Ap, k  , avlen) ;                                 \
        pA_end   = GBP_A (Ap, k+1, avlen) ;                                 \
    }                                                                       \
    const int64_t aknz = pA_end - pA_start

//------------------------------------------------------------------------------
// GB_GET_M_ij: get the numeric value of M(i,j)
//------------------------------------------------------------------------------

#define GB_GET_M_ij(pM)                             \
    /* get M(i,j), at Mi [pM] and Mx [pM] */        \
    bool mij = GBB_M (Mb, pM) && GB_MCAST (Mx, pM, msize)

//------------------------------------------------------------------------------
// GB_MULT_A_ik_B_kj: declare t and compute t = A(i,k) * B(k,j)
//------------------------------------------------------------------------------

#if ( GB_IS_PAIR_MULTIPLIER && !GB_Z_IS_COMPLEX )

    // PAIR multiplier: t is always 1; no numeric work to do to compute t.
    // The LXOR_PAIR and PLUS_PAIR semirings need the value t = 1 to use in
    // their monoid operator, however.
    #define t GB_PAIR_ONE
    #define GB_MULT_A_ik_B_kj

#elif ( GB_IS_FIRSTJ_MULTIPLIER || GB_IS_SECONDJ_MULTIPLIER )

    // nothing to do; t = aik*bkj already defined in an outer loop
    #define GB_MULT_A_ik_B_kj

#else

    // typical semiring
    #define GB_MULT_A_ik_B_kj                                       \
        GB_DECLAREA (aik) ;                                         \
        GB_GETA (aik, Ax, pA, A_iso) ;  /* aik = Ax [pA] ;  */      \
        GB_CIJ_DECLARE (t) ;            /* ctype t ;        */      \
        GB_MULT (t, aik, bkj, i, k, j)  /* t = aik * bkj ;  */

#endif

//------------------------------------------------------------------------------
// GB_GATHER_ALL_C_j: gather the values and pattern of C(:,j)
//------------------------------------------------------------------------------

// gather the pattern and values of C(:,j) for a coarse Gustavson task;
// the pattern is not flagged as jumbled.

#if GB_IS_ANY_PAIR_SEMIRING

    // ANY_PAIR: result is purely symbolic; no numeric work to do
    #define GB_GATHER_ALL_C_j(mark)                                 \
        for (int64_t i = 0 ; i < cvlen ; i++)                       \
        {                                                           \
            if (Hf [i] == mark)                                     \
            {                                                       \
                Ci [pC++] = i ;                                     \
            }                                                       \
        }

#else

    // typical semiring
    #define GB_GATHER_ALL_C_j(mark)                                 \
        for (int64_t i = 0 ; i < cvlen ; i++)                       \
        {                                                           \
            if (Hf [i] == mark)                                     \
            {                                                       \
                GB_CIJ_GATHER (pC, i) ; /* Cx [pC] = Hx [i] */      \
                Ci [pC++] = i ;                                     \
            }                                                       \
        }

#endif

//------------------------------------------------------------------------------
// GB_SORT_C_j_PATTERN: sort C(:,j) for a coarse task, or flag as jumbled
//------------------------------------------------------------------------------

// Only coarse tasks do the optional sort.  Fine hash tasks always leave C
// jumbled.

#define GB_SORT_C_j_PATTERN                                     \
    if (do_sort)                                                \
    {                                                           \
        /* sort the pattern of C(:,j) (non-default) */          \
        GB_qsort_1 (Ci + Cp [kk], cjnz) ;                       \
    }                                                           \
    else                                                        \
    {                                                           \
        /* lazy sort: C(:,j) is now jumbled (default) */        \
        task_C_jumbled = true ;                                 \
    }

//------------------------------------------------------------------------------
// GB_SORT_AND_GATHER_C_j: sort and gather C(:,j) for a coarse Gustavson task
//------------------------------------------------------------------------------

// gather the values of C(:,j) for a coarse Gustavson task
#if GB_IS_ANY_PAIR_SEMIRING

    // ANY_PAIR: result is purely symbolic
    #define GB_SORT_AND_GATHER_C_j                              \
        GB_SORT_C_j_PATTERN ;

#else

    // typical semiring
    #define GB_SORT_AND_GATHER_C_j                              \
        GB_SORT_C_j_PATTERN ;                                   \
        /* gather the values into C(:,j) */                     \
        for (int64_t pC = Cp [kk] ; pC < Cp [kk+1] ; pC++)      \
        {                                                       \
            const int64_t i = Ci [pC] ;                         \
            GB_CIJ_GATHER (pC, i) ;   /* Cx [pC] = Hx [i] */    \
        }

#endif

//------------------------------------------------------------------------------
// GB_SORT_AND_GATHER_HASHED_C_j: sort and gather C(:,j) for a coarse hash task
//------------------------------------------------------------------------------

#if GB_IS_ANY_PAIR_SEMIRING

    // ANY_PAIR: result is purely symbolic
    #define GB_SORT_AND_GATHER_HASHED_C_j(hash_mark)                    \
        GB_SORT_C_j_PATTERN ;

#else

    // gather the values of C(:,j) for a coarse hash task
    #define GB_SORT_AND_GATHER_HASHED_C_j(hash_mark)                    \
        GB_SORT_C_j_PATTERN ;                                           \
        for (int64_t pC = Cp [kk] ; pC < Cp [kk+1] ; pC++)              \
        {                                                               \
            const int64_t i = Ci [pC] ;                                 \
            for (GB_HASH (i))           /* find i in hash table */      \
            {                                                           \
                if (Hf [hash] == (hash_mark) && (Hi [hash] == i))       \
                {                                                       \
                    /* i found in the hash table */                     \
                    /* Cx [pC] = Hx [hash] ; */                         \
                    GB_CIJ_GATHER (pC, hash) ;                          \
                    break ;                                             \
                }                                                       \
            }                                                           \
        }

#endif

//------------------------------------------------------------------------------
// GB_Z_ATOMIC_UPDATE_HX:  Hx [i] += t
//------------------------------------------------------------------------------

#if GB_IS_ANY_MONOID

    //--------------------------------------------------------------------------
    // The update Hx [i] += t can be skipped entirely, for the ANY monoid.
    //--------------------------------------------------------------------------

    #define GB_Z_ATOMIC_UPDATE_HX(i,t)

#elif GB_Z_HAS_ATOMIC_UPDATE

    //--------------------------------------------------------------------------
    // Hx [i] += t via atomic update
    //--------------------------------------------------------------------------

    // for built-in MIN/MAX monoids only, on built-in types
    #define GB_MINMAX(i,t,done)                                     \
    {                                                               \
        GB_Z_TYPE zold, znew, *pz = Hx + (i) ;                      \
        do                                                          \
        {                                                           \
            /* zold = Hx [i] via atomic read */                     \
            GB_Z_ATOMIC_READ (zold, *pz) ;                          \
            /* done if zold <= t for MIN, or zold >= t for MAX, */  \
            /* but not done if zold is NaN */                       \
            if (done) break ;                                       \
            znew = t ;  /* t should be assigned; it is not NaN */   \
        }                                                           \
        while (!GB_Z_ATOMIC_COMPARE_EXCHANGE (pz, zold, znew)) ;    \
    }

    #if GB_IS_IMIN_MONOID

        // built-in MIN monoids for signed and unsigned integers
        #define GB_Z_ATOMIC_UPDATE_HX(i,t) GB_MINMAX (i, t, zold <= t)

    #elif GB_IS_IMAX_MONOID

        // built-in MAX monoids for signed and unsigned integers
        #define GB_Z_ATOMIC_UPDATE_HX(i,t) GB_MINMAX (i, t, zold >= t)

    #elif GB_IS_FMIN_MONOID

        // built-in MIN monoids for float and double, with omitnan behavior.
        // The update is skipped entirely if t is NaN.  Otherwise, if t is not
        // NaN, zold is checked.  If zold is NaN, islessequal (zold, t) is
        // always false, so the non-NaN t must be always be assigned to Hx [i].
        // If both terms are not NaN, then islessequal (zold,t) is just
        // zold <= t.  If that is true, there is no work to do and
        // the loop breaks.  Otherwise, t is smaller than zold and so it must
        // be assigned to Hx [i].
        #define GB_Z_ATOMIC_UPDATE_HX(i,t)                          \
        {                                                           \
            if (!isnan (t))                                         \
            {                                                       \
                GB_MINMAX (i, t, islessequal (zold, t)) ;           \
            }                                                       \
        }

    #elif GB_IS_FMAX_MONOID

        // built-in MAX monoids for float and double, with omitnan behavior.
        #define GB_Z_ATOMIC_UPDATE_HX(i,t)                          \
        {                                                           \
            if (!isnan (t))                                         \
            {                                                       \
                GB_MINMAX (i, t, isgreaterequal (zold, t)) ;        \
            }                                                       \
        }

    #elif GB_IS_PLUS_FC32_MONOID

        // built-in PLUS_FC32 monoid can be done as two independent atomics
        #define GB_Z_ATOMIC_UPDATE_HX(i,t)                          \
        {                                                           \
            GB_ATOMIC_UPDATE                                        \
            Hx_real [2*(i)] += GB_crealf (t) ;                      \
            GB_ATOMIC_UPDATE                                        \
            Hx_imag [2*(i)] += GB_cimagf (t) ;                      \
        }

    #elif GB_IS_PLUS_FC64_MONOID

        // built-in PLUS_FC64 monoid can be done as two independent atomics
        #define GB_Z_ATOMIC_UPDATE_HX(i,t)                          \
        {                                                           \
            GB_ATOMIC_UPDATE                                        \
            Hx_real [2*(i)] += GB_creal (t) ;                       \
            GB_ATOMIC_UPDATE                                        \
            Hx_imag [2*(i)] += GB_cimag (t) ;                       \
        }

    #elif GB_Z_HAS_OMP_ATOMIC_UPDATE

        // built-in PLUS and TIMES for integers and real, and boolean LOR,
        // LAND, LXOR monoids can be implemented with an OpenMP pragma.
        #define GB_Z_ATOMIC_UPDATE_HX(i,t)                          \
        {                                                           \
            GB_ATOMIC_UPDATE                                        \
            GB_HX_UPDATE (i, t) ;                                   \
        }

    #else

        // all other atomic monoids (EQ, XNOR) on boolean, signed and unsigned
        // integers, float, double, and float complex (not double complex).
        // user-defined monoids can use this if zsize is 1, 2, 4, or 8.
        #define GB_Z_ATOMIC_UPDATE_HX(i,t)                              \
        {                                                               \
            GB_Z_TYPE zold, znew, *pz = Hx + (i) ;                      \
            do                                                          \
            {                                                           \
                /* zold = Hx [i] via atomic read */                     \
                GB_Z_ATOMIC_READ (zold, *pz) ;                          \
                /* znew = zold + t */                                   \
                GB_ADD (znew, zold, t) ;                                \
            }                                                           \
            while (!GB_Z_ATOMIC_COMPARE_EXCHANGE (pz, zold, znew)) ;    \
        }

    #endif

#else

    //--------------------------------------------------------------------------
    // Hx [i] += t can only be done inside the critical section
    //--------------------------------------------------------------------------

    // all user-defined monoids go here, and all double complex monoids (except
    // PLUS).  This macro is not actually atomic itself, but must be placed
    // inside a critical section.
    #define GB_Z_ATOMIC_UPDATE_HX(i,t)  \
    {                                   \
        GB_OMP_FLUSH                    \
        GB_HX_UPDATE (i, t) ;           \
        GB_OMP_FLUSH                    \
    }

#endif

#define GB_IS_MINMAX_MONOID \
    (GB_IS_IMIN_MONOID || GB_IS_IMAX_MONOID ||  \
     GB_IS_FMIN_MONOID || GB_IS_FMAX_MONOID)

//------------------------------------------------------------------------------
// GB_Z_ATOMIC_WRITE_HX:  Hx [i] = t via atomics or critical section
//------------------------------------------------------------------------------

#if GB_Z_HAS_ATOMIC_WRITE

    //--------------------------------------------------------------------------
    // Hx [i] = t via atomic write
    //--------------------------------------------------------------------------

    // The GB_Z_TYPE has an atomic write, with GB_Z_ATOMIIC_BITS defined.
    // The generic methods do not use this option.  For the ANY_PAIR semiring,
    // GB_Z_ATOMIC_BITS is zero and this macro becomes empty.
    #define GB_Z_ATOMIC_WRITE_HX(i,t) GB_Z_ATOMIC_WRITE (Hx [i], t)

#else

    //--------------------------------------------------------------------------
    // Hx [i] = t via critical section
    //--------------------------------------------------------------------------

    // This macro is not actually atomic itself, but must be placed inside a
    // critical section.  GB_HX_WRITE is a memcpy for generic methods.
    #define GB_Z_ATOMIC_WRITE_HX(i,t)       \
    {                                       \
        GB_OMP_FLUSH                        \
        GB_HX_WRITE (i, t) ;                \
        GB_OMP_FLUSH                        \
    }

#endif

//------------------------------------------------------------------------------
// hash iteration
//------------------------------------------------------------------------------

// to iterate over the hash table, looking for index i:
// 
//      for (GB_HASH (i))
//      {
//          ...
//      }
//
// which expands into the following, where f(i) is the GB_HASHF(i) hash
// function:
//
//      for (int64_t hash = f(i) ; ; hash = (hash+1)&(hash_size-1))
//      {
//          ...
//      }

#define GB_HASH(i) \
    int64_t hash = GB_HASHF (i,hash_bits) ; ; GB_REHASH (hash,i,hash_bits)

//------------------------------------------------------------------------------
// define macros for any sparsity of A and B
//------------------------------------------------------------------------------

#ifdef GB_JIT_KERNEL
// JIT kernels keep their specializations for the sparsity of A and B
#define GB_META16
#else
// generic and pre-generated: define macros for any sparsity of A and B
#undef GB_META16
#endif

#include "GB_meta16_definitions.h"

#endif

