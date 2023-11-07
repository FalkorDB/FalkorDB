//------------------------------------------------------------------------------
// GB_convert_bitmap_worker: construct triplets or CSC/CSR from bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: needed.

// If A is iso and Ax_new is not NULL, the iso scalar is expanded into the
// non-iso array Ax_new.  Otherwise, if Ax_new and Ax are NULL then no values
// are extracted.

// TODO allow this function to do typecasting.

#include "GB.h"
#include "GB_partition.h"
#include "GB_unused.h"

GrB_Info GB_convert_bitmap_worker   // extract CSC/CSR or triplets from bitmap
(
    // outputs:
    int64_t *restrict Ap,           // vector pointers for CSC/CSR form
    int64_t *restrict Ai,           // indices for CSC/CSR or triplet form
    int64_t *restrict Aj,           // vector indices for triplet form
    GB_void *restrict Ax_new,       // values for CSC/CSR or triplet form
    int64_t *anvec_nonempty,        // # of non-empty vectors
    // inputs: not modified
    const GrB_Matrix A,             // matrix to extract; not modified
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (GB_IS_BITMAP (A)) ;
    ASSERT (Ap != NULL) ;           // must be provided on input, size avdim+1

    int64_t *restrict W = NULL ; size_t W_size = 0 ;
    const int64_t avdim = A->vdim ;
    const int64_t avlen = A->vlen ;
    const size_t asize = A->type->size ;

    //--------------------------------------------------------------------------
    // count the entries in each vector
    //--------------------------------------------------------------------------

    const int8_t *restrict Ab = A->b ;

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (avlen*avdim, chunk, nthreads_max) ;
    bool by_vector = (nthreads <= avdim) ;

    if (by_vector)
    {

        //----------------------------------------------------------------------
        // compute all vectors in parallel (no workspace)
        //----------------------------------------------------------------------

        int64_t j ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (j = 0 ; j < avdim ; j++)
        {
            // ajnz = nnz (A (:,j))
            int64_t ajnz = 0 ;
            int64_t pA_start = j * avlen ;
            for (int64_t i = 0 ; i < avlen ; i++)
            { 
                // see if A(i,j) is present in the bitmap
                int64_t p = i + pA_start ;
                ajnz += Ab [p] ;
                ASSERT (Ab [p] == 0 || Ab [p] == 1) ;
            }
            Ap [j] = ajnz ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // compute blocks of rows in parallel
        //----------------------------------------------------------------------

        // allocate one row of W per thread, each row of length avdim
        W = GB_MALLOC_WORK (nthreads * avdim, int64_t, &W_size) ;
        if (W == NULL)
        {
            // out of memory
            return (GrB_OUT_OF_MEMORY) ;
        }

        int taskid ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (taskid = 0 ; taskid < nthreads ; taskid++)
        {
            int64_t *restrict Wtask = W + taskid * avdim ;
            int64_t istart, iend ;
            GB_PARTITION (istart, iend, avlen, taskid, nthreads) ;
            for (int64_t j = 0 ; j < avdim ; j++)
            {
                // ajnz = nnz (A (istart:iend-1,j))
                int64_t ajnz = 0 ;
                int64_t pA_start = j * avlen ;
                for (int64_t i = istart ; i < iend ; i++)
                { 
                    // see if A(i,j) is present in the bitmap
                    int64_t p = i + pA_start ;
                    ajnz += Ab [p] ;
                    ASSERT (Ab [p] == 0 || Ab [p] == 1) ;
                }
                Wtask [j] = ajnz ;
            }
        }

        // cumulative sum to compute nnz(A(:,j)) for each vector j
        int64_t j ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (j = 0 ; j < avdim ; j++)
        {
            int64_t ajnz = 0 ;
            for (int taskid = 0 ; taskid < nthreads ; taskid++)
            { 
                int64_t *restrict Wtask = W + taskid * avdim ;
                int64_t c = Wtask [j] ;
                Wtask [j] = ajnz ;
                ajnz += c ;
            }
            Ap [j] = ajnz ;
        }
    }

    //--------------------------------------------------------------------------
    // cumulative sum of Ap 
    //--------------------------------------------------------------------------

    int nth = GB_nthreads (avdim, chunk, nthreads_max) ;
    GB_cumsum (Ap, avdim, anvec_nonempty, nth, Werk) ;
    ASSERT (Ap [avdim] == A->nvals) ;

    //--------------------------------------------------------------------------
    // gather the pattern and values from the bitmap
    //--------------------------------------------------------------------------

    // TODO: add type-specific versions for built-in types

    const GB_void *restrict Ax = (GB_void *) (A->x) ;
    const bool A_iso = A->iso ;
    const bool numeric = (Ax_new != NULL && Ax != NULL) ;

    if (by_vector)
    {

        //----------------------------------------------------------------------
        // construct all vectors in parallel (no workspace)
        //----------------------------------------------------------------------

        int64_t j ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (j = 0 ; j < avdim ; j++)
        {
            // gather from the bitmap into the new A (:,j)
            int64_t pnew = Ap [j] ;
            int64_t pA_start = j * avlen ;
            for (int64_t i = 0 ; i < avlen ; i++)
            {
                int64_t p = i + pA_start ;
                if (Ab [p])
                {
                    // A(i,j) is in the bitmap
                    if (Ai != NULL) Ai [pnew] = i ;
                    if (Aj != NULL) Aj [pnew] = j ;
                    if (numeric)
                    { 
                        // Ax_new [pnew] = Ax [p])
                        memcpy (Ax_new +(pnew)*asize,
                            Ax +(A_iso ? 0:(p)*asize), asize) ;
                    }
                    pnew++ ;
                }
            }
            ASSERT (pnew == Ap [j+1]) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // compute blocks of rows in parallel
        //----------------------------------------------------------------------

        int taskid ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (taskid = 0 ; taskid < nthreads ; taskid++)
        {
            int64_t *restrict Wtask = W + taskid * avdim ;
            int64_t istart, iend ;
            GB_PARTITION (istart, iend, avlen, taskid, nthreads) ;
            for (int64_t j = 0 ; j < avdim ; j++)
            {
                // gather from the bitmap into the new A (:,j)
                int64_t pnew = Ap [j] + Wtask [j] ;
                int64_t pA_start = j * avlen ;
                for (int64_t i = istart ; i < iend ; i++)
                {
                    // see if A(i,j) is present in the bitmap
                    int64_t p = i + pA_start ;
                    if (Ab [p])
                    {
                        // A(i,j) is in the bitmap
                        if (Ai != NULL) Ai [pnew] = i ;
                        if (Aj != NULL) Aj [pnew] = j ;
                        if (numeric)
                        { 
                            // Ax_new [pnew] = Ax [p] ;
                            memcpy (Ax_new +(pnew)*asize,
                                Ax +(A_iso ? 0:(p)*asize), asize) ;
                        }
                        pnew++ ;
                    }
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // free workspace return result
    //--------------------------------------------------------------------------

    GB_FREE_WORK (&W, W_size) ;
    return (GrB_SUCCESS) ;
}

