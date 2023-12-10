//------------------------------------------------------------------------------
// GB_emult_02_phase1: C = A.*B where A is sparse/hyper and B is bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed, but could use one for each mask type.

// Symbolic analysis phase for GB_emult_02 and GB_emult_03.

#define GB_FREE_ALL ;

#include "GB_ewise.h"
#include "GB_emult.h"
#include "GB_binop.h"
#include "GB_stringify.h"

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
    int64_t *restrict Wfirst,
    int64_t *restrict Wlast,
    // output:
    int64_t *Cp_kfirst,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // get C, M, A, and B
    //--------------------------------------------------------------------------

    GrB_Info info ;
    const int8_t  *restrict Mb = (M == NULL) ? NULL : M->b ;
    const GB_M_TYPE *restrict Mx = (M == NULL || Mask_struct) ? NULL :
        (const GB_M_TYPE *) M->x ;
    const size_t msize = (M == NULL) ? 0 : M->type->size ;

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int64_t *restrict Ai = A->i ;
    const int64_t vlen = A->vlen ;
    const int64_t nvec = A->nvec ;
    const int64_t anz = GB_nnz (A) ;

    const int8_t *restrict Bb = B->b ;
    const bool B_is_bitmap = GB_IS_BITMAP (B) ;

    int64_t *restrict Cp = C->p ;

    const int64_t *restrict kfirst_Aslice = A_ek_slicing ;
    const int64_t *restrict klast_Aslice  = A_ek_slicing + A_ntasks ;
    const int64_t *restrict pstart_Aslice = A_ek_slicing + A_ntasks * 2 ;

    //--------------------------------------------------------------------------
    // count entries in C
    //--------------------------------------------------------------------------

    C->nvec_nonempty = A->nvec_nonempty ;
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
                    int64_t j = GBH (Ah, k) ;
                    int64_t pB_start = j * vlen ;
                    int64_t pA, pA_end ;
                    GB_get_pA (&pA, &pA_end, tid, k,
                        kfirst, klast, pstart_Aslice, Ap, vlen) ;
                    int64_t cjnz = 0 ;
                    for ( ; pA < pA_end ; pA++)
                    { 
                        cjnz += Bb [pB_start + Ai [pA]] ;
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
                        Cp [k] = cjnz ; 
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
                    int64_t j = GBH (Ah, k) ;
                    int64_t pB_start = j * vlen ;
                    int64_t pA, pA_end ;
                    GB_get_pA (&pA, &pA_end, tid, k,
                        kfirst, klast, pstart_Aslice, Ap, vlen) ;
                    int64_t cjnz = 0 ;
                    for ( ; pA < pA_end ; pA++)
                    { 
                        int64_t i = Ai [pA] ;
                        int64_t pB = pB_start + i ;
                        bool mij = GBB (Mb, pB) && GB_MCAST (Mx, pB, msize) ;
                        mij = mij ^ Mask_comp ;
                        cjnz += (mij && GBB (Bb, pB)) ;
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
                        Cp [k] = cjnz ; 
                    }
                }
            }
        }

        //----------------------------------------------------------------------
        // finalize Cp, cumulative sum of Cp and compute Cp_kfirst
        //----------------------------------------------------------------------

        GB_ek_slice_merge1 (Cp, Wfirst, Wlast, A_ek_slicing, A_ntasks) ;
        GB_ek_slice_merge2 (&(C->nvec_nonempty), Cp_kfirst, Cp, nvec,
            Wfirst, Wlast, A_ek_slicing, A_ntasks, A_nthreads, Werk) ;
    }

    //--------------------------------------------------------------------------
    // allocate C->i and C->x
    //--------------------------------------------------------------------------

    int64_t cnz = (C_has_pattern_of_A) ? anz : Cp [nvec] ;
    // set C->iso = C_iso   OK
    GB_OK (GB_bix_alloc (C, cnz, GxB_SPARSE, false, true, C_iso)) ;

    //--------------------------------------------------------------------------
    // copy pattern into C
    //--------------------------------------------------------------------------

    // TODO: could make these components of C shallow instead of memcpy

    if (GB_IS_HYPERSPARSE (A))
    { 
        // copy A->h into C->h
        GB_memcpy (C->h, Ah, nvec * sizeof (int64_t), A_nthreads) ;
    }

    if (C_has_pattern_of_A)
    { 
        // Method2/3(b): B is full and no mask present, so the pattern of C is
        // the same as the pattern of A
        GB_memcpy (Cp, Ap, (nvec+1) * sizeof (int64_t), A_nthreads) ;
        GB_memcpy (C->i, Ai, cnz * sizeof (int64_t), A_nthreads) ;
    }

    C->nvals = cnz ;
    C->jumbled = A->jumbled ;
    C->magic = GB_MAGIC ;

    return (GrB_SUCCESS) ;
}


