//------------------------------------------------------------------------------
// GB_emult_02_phase1: C = A.*B where A is sparse/hyper and B is bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed: factory cases: mask types, M bitmap/full, B bitmap/full,
// A sparse/hyper

// Symbolic analysis phase for GB_emult_02 and GB_emult_03.

#define GB_FREE_ALL ;

#include "ewise/GB_ewise.h"
#include "emult/GB_emult.h"
#include "binaryop/GB_binop.h"
#include "jitifyer/GB_stringify.h"
#include "slice/factory/GB_ek_slice_merge.h"

GrB_Info GB_emult_02_phase1 // symbolic analysis for GB_emult_02 and GB_emult_03
(
    // input/output:
    GrB_Matrix C,
    // input:
    const bool C_iso,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int64_t *restrict A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads,
    // workspace:
    uint64_t *restrict Wfirst,
    uint64_t *restrict Wlast,
    // output:
    uint64_t *Cp_kfirst,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // get C, M, A, and B
    //--------------------------------------------------------------------------

    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;
    ASSERT (GB_IS_BITMAP (B) || GB_IS_FULL (B)) ;
    ASSERT ((M == NULL) || GB_IS_BITMAP (M) || GB_IS_FULL (M)) ;

    GrB_Info info ;
    const int8_t  *restrict Mb = (M == NULL) ? NULL : M->b ;
    const GB_M_TYPE *restrict Mx = (M == NULL || Mask_struct) ? NULL :
        (const GB_M_TYPE *) M->x ;
    const size_t msize = (M == NULL) ? 0 : M->type->size ;

    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
    GB_Ai_DECLARE (Ai, const) ; GB_Ai_PTR (Ai, A) ;

    const int64_t vlen = A->vlen ;
    const int64_t nvec = A->nvec ;
    const int64_t anz = GB_nnz (A) ;

    const int8_t *restrict Bb = B->b ;
    const bool B_is_bitmap = GB_IS_BITMAP (B) ;

    GB_Cp_DECLARE (Cp, ) ; GB_Cp_PTR (Cp, C) ;
    const bool Cp_is_32 = C->p_is_32 ;
    const bool Cj_is_32 = C->j_is_32 ;
    const bool Ci_is_32 = C->i_is_32 ;

    ASSERT (C->p_is_32 == A->p_is_32) ;
    ASSERT (C->j_is_32 == A->j_is_32) ;
    ASSERT (C->i_is_32 == A->i_is_32) ;

    const int64_t *restrict kfirst_Aslice = A_ek_slicing ;
    const int64_t *restrict klast_Aslice  = A_ek_slicing + A_ntasks ;
    const int64_t *restrict pstart_Aslice = A_ek_slicing + A_ntasks * 2 ;

    //--------------------------------------------------------------------------
    // count entries in C
    //--------------------------------------------------------------------------

//  C->nvec_nonempty = A->nvec_nonempty ;
    GB_nvec_nonempty_set (C, GB_nvec_nonempty_get (A)) ;
    C->nvec = nvec ;
    const bool C_has_pattern_of_A = !B_is_bitmap && (M == NULL) ;

    if (!C_has_pattern_of_A)
    { 

        // This phase is very similar to GB_select_entry_phase1_template.c.

        if (M == NULL)
        {

            //------------------------------------------------------------------
            // Method2/3(a): C = A.*B where A is sparse/hyper and B is bitmap
            //------------------------------------------------------------------

            ASSERT (GB_IS_BITMAP (B)) ;

            int tid ;
            #pragma omp parallel for num_threads(A_nthreads) schedule(dynamic,1)
            for (tid = 0 ; tid < A_ntasks ; tid++)
            {
                int64_t kfirst = kfirst_Aslice [tid] ;
                int64_t klast  = klast_Aslice  [tid] ;
                Wfirst [tid] = 0 ;
                Wlast  [tid] = 0 ;
                for (int64_t k = kfirst ; k <= klast ; k++)
                {
                    // count the entries in C(:,j)
                    int64_t j = GBh_A (Ah, k) ;
                    int64_t pB_start = j * vlen ;
                    GB_GET_PA (pA, pA_end, tid, k, kfirst, klast, pstart_Aslice,
                        GB_IGET (Ap, k), GB_IGET (Ap, k+1)) ;
                    int64_t cjnz = 0 ;
                    for ( ; pA < pA_end ; pA++)
                    { 
                        cjnz += Bb [pB_start + GB_IGET (Ai, pA)] ;
                    }
                    if (k == kfirst)
                    { 
                        Wfirst [tid] = cjnz ;
                    }
                    else if (k == klast)
                    { 
                        Wlast [tid] = cjnz ;
                    }
                    else
                    { 
                        GB_ISET (Cp, k, cjnz) ;     // Cp [k] = cjnz ;
                    }
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // Method2/3(c): C<#M> = A.*B; A is sparse/hyper; M, B bitmap/full
            //------------------------------------------------------------------

            ASSERT (M != NULL) ;
            ASSERT (GB_IS_BITMAP (M) || GB_IS_FULL (M)) ;
            ASSERT (GB_IS_BITMAP (B) || GB_IS_FULL (B)) ;

            int tid ;
            #pragma omp parallel for num_threads(A_nthreads) schedule(dynamic,1)
            for (tid = 0 ; tid < A_ntasks ; tid++)
            {
                int64_t kfirst = kfirst_Aslice [tid] ;
                int64_t klast  = klast_Aslice  [tid] ;
                Wfirst [tid] = 0 ;
                Wlast  [tid] = 0 ;
                for (int64_t k = kfirst ; k <= klast ; k++)
                {
                    // count the entries in C(:,j)
                    int64_t j = GBh_A (Ah, k) ;
                    int64_t pB_start = j * vlen ;
                    GB_GET_PA (pA, pA_end, tid, k, kfirst, klast, pstart_Aslice,
                        GB_IGET (Ap, k), GB_IGET (Ap, k+1)) ;
                    int64_t cjnz = 0 ;
                    for ( ; pA < pA_end ; pA++)
                    { 
                        int64_t i = GB_IGET (Ai, pA) ;
                        int64_t pB = pB_start + i ;
                        bool mij = GBb_M (Mb, pB) && GB_MCAST (Mx, pB, msize) ;
                        mij = mij ^ Mask_comp ;
                        cjnz += (mij && GBb_M (Bb, pB)) ;
                    }
                    if (k == kfirst)
                    { 
                        Wfirst [tid] = cjnz ;
                    }
                    else if (k == klast)
                    { 
                        Wlast [tid] = cjnz ;
                    }
                    else
                    { 
                        GB_ISET (Cp, k, cjnz) ;     // Cp [k] = cjnz ;
                    }
                }
            }
        }

        //----------------------------------------------------------------------
        // finalize Cp, cumulative sum of Cp and compute Cp_kfirst
        //----------------------------------------------------------------------

        GB_ek_slice_merge1 (Cp, Cp_is_32,
            Wfirst, Wlast, A_ek_slicing, A_ntasks) ;

        int64_t nvec_nonempty ;
        GB_cumsum (Cp, Cp_is_32, nvec, &nvec_nonempty, A_nthreads, Werk) ;
        GB_nvec_nonempty_set (C, nvec_nonempty) ;

        GB_ek_slice_merge2 (Cp_kfirst, Cp, Cp_is_32,
            Wfirst, Wlast, A_ek_slicing, A_ntasks) ;
    }

    //--------------------------------------------------------------------------
    // allocate C->i and C->x
    //--------------------------------------------------------------------------

    int64_t cnz = (C_has_pattern_of_A) ? anz : GB_IGET (Cp, nvec) ;
    GB_OK (GB_bix_alloc (C, cnz, GxB_SPARSE, false, true, C_iso)) ;

    //--------------------------------------------------------------------------
    // copy pattern into C
    //--------------------------------------------------------------------------

    // FUTURE: could make these components of C shallow instead of memcpy

    size_t cpsize = Cp_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t cjsize = Cj_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t cisize = Ci_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;

    if (GB_IS_HYPERSPARSE (A))
    { 
        // copy A->h into C->h
        GB_memcpy (C->h, Ah, nvec * cjsize, A_nthreads) ;
    }

    if (C_has_pattern_of_A)
    { 
        // Method2/3(b): B is full and no mask present, so the pattern of C is
        // the same as the pattern of A
        GB_memcpy (Cp, Ap, (nvec+1) * cpsize, A_nthreads) ;
        GB_memcpy (C->i, Ai, cnz * cisize, A_nthreads) ;
    }

    C->nvals = cnz ;
    C->jumbled = A->jumbled ;
    C->magic = GB_MAGIC ;

    return (GrB_SUCCESS) ;
}


