//------------------------------------------------------------------------------
// GB_subassign_23_template: C += A where C is full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_unused.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL                         \
{                                           \
    GB_WERK_POP (A_ek_slicing, int64_t) ;   \
}

{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_KERNEL
    #define A_is_bitmap GB_A_IS_BITMAP
    #define A_is_full   GB_A_IS_FULL
    #define A_iso       GB_A_ISO
    #else
    bool A_is_bitmap = GB_IS_BITMAP (A) ;
    bool A_is_full = GB_IS_FULL (A) ;
    const bool A_iso = A->iso ;
    #endif

    //--------------------------------------------------------------------------
    // slice the A matrix
    //--------------------------------------------------------------------------

    GB_WERK_DECLARE (A_ek_slicing, int64_t) ;
    int A_ntasks, A_nthreads ;
    GB_A_NHELD (anz) ;      // int64_t anz = GB_nnz_held (A) ;
    double work = anz + A->nvec ;
    if (A_is_bitmap || A_is_full)
    { 
        // C is full and A is bitmap or full
        A_nthreads = GB_nthreads (work, chunk, nthreads_max) ;
        A_ntasks = 0 ;   // unused
        ASSERT (A_ek_slicing == NULL) ;
    }
    else
    { 
        // create tasks to compute over the matrix A
        GB_SLICE_MATRIX_WORK (A, 32, work, anz) ;
        ASSERT (A_ek_slicing != NULL) ;
    }

    //--------------------------------------------------------------------------
    // get C and A
    //--------------------------------------------------------------------------

    ASSERT (!C->iso) ;
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    ASSERT (GB_IS_FULL (C)) ;
    GB_C_NHELD (cnz) ;      // const int64_t cnz = GB_nnz_held (C) ;
    GB_DECLAREY (ywork) ;
    if (A_iso)
    { 
        // get the iso value of A and typecast it to Y
        // ywork = (ytype) Ax [0]
        GB_COPY_aij_to_ywork (ywork, Ax, 0, true) ;
    }

    if (A_is_bitmap)
    {

        //----------------------------------------------------------------------
        // C += A when C is full and A is bitmap
        //----------------------------------------------------------------------

        const int8_t *restrict Ab = A->b ;
        int64_t p ;
        #pragma omp parallel for num_threads(A_nthreads) schedule(static)
        for (p = 0 ; p < cnz ; p++)
        { 
            if (!Ab [p]) continue ;
            // Cx [p] += (ytype) Ax [p], with typecasting
            GB_ACCUMULATE_aij (Cx, p, Ax, p, A_iso, ywork) ;
        }

    }
    else if (A_is_full)
    {

        //----------------------------------------------------------------------
        // C += A when both C and A are ffull
        //----------------------------------------------------------------------

        int64_t p ;
        #pragma omp parallel for num_threads(A_nthreads) schedule(static)
        for (p = 0 ; p < cnz ; p++)
        { 
            // Cx [p] += (ytype) Ax [p], with typecasting
            GB_ACCUMULATE_aij (Cx, p, Ax, p, A_iso, ywork) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // C += A when C is full and A is sparse
        //----------------------------------------------------------------------

        ASSERT (GB_JUMBLED_OK (A)) ;

        const int64_t *restrict Ap = A->p ;
        const int64_t *restrict Ah = A->h ;
        const int64_t *restrict Ai = A->i ;
        const int64_t avlen = A->vlen ;
        const int64_t cvlen = C->vlen ;
        bool A_jumbled = A->jumbled ;

        const int64_t *restrict kfirst_Aslice = A_ek_slicing ;
        const int64_t *restrict klast_Aslice  = kfirst_Aslice + A_ntasks ;
        const int64_t *restrict pstart_Aslice = klast_Aslice + A_ntasks ;

        int taskid ;
        #pragma omp parallel for num_threads(A_nthreads) schedule(dynamic,1)
        for (taskid = 0 ; taskid < A_ntasks ; taskid++)
        {

            // if kfirst > klast then taskid does no work at all
            int64_t kfirst = kfirst_Aslice [taskid] ;
            int64_t klast  = klast_Aslice  [taskid] ;

            //------------------------------------------------------------------
            // C(:,kfirst:klast) += A(:,kfirst:klast)
            //------------------------------------------------------------------

            for (int64_t k = kfirst ; k <= klast ; k++)
            {

                //--------------------------------------------------------------
                // find the part of A(:,k) and C(:,k) for this task
                //--------------------------------------------------------------

                int64_t j = GBH_A (Ah, k) ;
                int64_t pA_start = GBP_A (Ap, k, avlen) ;
                int64_t pA_end   = GBP_A (Ap, k+1, avlen) ;
                GB_GET_PA (my_pA_start, my_pA_end, taskid, k,
                    kfirst, klast, pstart_Aslice, pA_start, pA_end) ;
                bool ajdense = ((pA_end - pA_start) == cvlen) ;

                // pC points to the start of C(:,j)
                int64_t pC = j * cvlen ;

                //--------------------------------------------------------------
                // C(:,j) += A(:,j)
                //--------------------------------------------------------------

                if (ajdense && !A_jumbled)
                {

                    //----------------------------------------------------------
                    // A(:,j) is dense
                    //----------------------------------------------------------

                    GB_PRAGMA_SIMD_VECTORIZE
                    for (int64_t pA = my_pA_start ; pA < my_pA_end ; pA++)
                    { 
                        int64_t i = pA - pA_start ;
                        int64_t p = pC + i ;
                        // Cx [p] += (ytype) Ax [pA], with typecasting
                        GB_ACCUMULATE_aij (Cx, p, Ax, pA, A_iso, ywork) ;
                    }

                }
                else
                {

                    //----------------------------------------------------------
                    // A(:,j) is sparse 
                    //----------------------------------------------------------

                    GB_PRAGMA_SIMD_VECTORIZE
                    for (int64_t pA = my_pA_start ; pA < my_pA_end ; pA++)
                    { 
                        int64_t i = Ai [pA] ;
                        int64_t p = pC + i ;
                        // Cx [p] += (ytype) Ax [pA], with typecasting
                        GB_ACCUMULATE_aij (Cx, p, Ax, pA, A_iso, ywork) ;
                    }
                }
            }
        }
    }

    GB_FREE_ALL ;
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL ;

