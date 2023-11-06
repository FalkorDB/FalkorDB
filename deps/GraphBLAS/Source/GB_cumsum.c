//------------------------------------------------------------------------------
// GB_cumsum: cumlative sum of an array
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed.  Only one variant possible.

// Compute the cumulative sum of an array count[0:n], of size n+1:

//      k = sum (count [0:n-1] != 0) ;
//      count = cumsum ([0 count[0:n-1]]) ;

// That is, count [j] on input is overwritten with sum (count [0..j-1]).
// On input, count [n] is not accessed and is implicitly zero on input.
// On output, count [n] is the total sum.

#include "GB.h"

void GB_cumsum                      // cumulative sum of an array
(
    int64_t *restrict count,     // size n+1, input/output
    const int64_t n,
    int64_t *restrict kresult,   // return k, if needed by the caller
    int nthreads,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (count != NULL) ;
    ASSERT (n >= 0) ;

    //--------------------------------------------------------------------------
    // determine # of threads to use
    //--------------------------------------------------------------------------

    #if !defined ( _OPENMP )
    nthreads = 1 ;
    #endif

    if (nthreads > 1)
    { 
        nthreads = GB_IMIN (nthreads, n / (64 * 1024)) ;
        nthreads = GB_IMAX (nthreads, 1) ;
    }

    //--------------------------------------------------------------------------
    // count = cumsum ([0 count[0:n-1]]) ;
    //--------------------------------------------------------------------------

    if (kresult == NULL)
    {

        if (nthreads <= 2)
        {

            //------------------------------------------------------------------
            // cumsum with one thread
            //------------------------------------------------------------------

            int64_t s = 0 ;
            for (int64_t i = 0 ; i < n ; i++)
            { 
                int64_t c = count [i] ;
                count [i] = s ;
                s += c ;
            }
            count [n] = s ;

        }
        else
        {

            //------------------------------------------------------------------
            // cumsum with multiple threads
            //------------------------------------------------------------------

            // allocate workspace
            GB_WERK_DECLARE (ws, int64_t) ;
            GB_WERK_PUSH (ws, nthreads, int64_t) ;
            if (ws == NULL)
            { 
                // out of memory; use a single thread instead
                GB_cumsum (count, n, NULL, 1, NULL) ;
                return ;
            }

            int tid ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (tid = 0 ; tid < nthreads ; tid++)
            {
                // each task sums up its own part
                int64_t istart, iend ;
                GB_PARTITION (istart, iend, n, tid, nthreads) ;
                int64_t s = 0 ;
                for (int64_t i = istart ; i < iend ; i++)
                { 
                    s += count [i] ;
                }
                ws [tid] = s ;
            }

            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (tid = 0 ; tid < nthreads ; tid++)
            {
                // each tasks computes the cumsum of its own part
                int64_t istart, iend ;
                GB_PARTITION (istart, iend, n, tid, nthreads) ;
                int64_t s = 0 ;
                for (int i = 0 ; i < tid ; i++)
                { 
                    s += ws [i] ;
                }
                for (int64_t i = istart ; i < iend ; i++)
                { 
                    int64_t c = count [i] ;
                    count [i] = s ;
                    s += c ;
                }
                if (iend == n)
                { 
                    count [n] = s ;
                }
            }

            // free workspace
            GB_WERK_POP (ws, int64_t) ;
        }

    }
    else
    {

        if (nthreads <= 2)
        {

            //------------------------------------------------------------------
            // cumsum with one thread, also compute k
            //------------------------------------------------------------------

            int64_t k = 0 ;
            int64_t s = 0 ;
            for (int64_t i = 0 ; i < n ; i++)
            { 
                int64_t c = count [i] ;
                if (c != 0) k++ ;
                count [i] = s ;
                s += c ;
            }
            count [n] = s ;
            (*kresult) = k ;

        }
        else
        {

            //------------------------------------------------------------------
            // cumsum with multiple threads, also compute k
            //------------------------------------------------------------------

            // allocate workspace
            GB_WERK_DECLARE (ws, int64_t) ;
            GB_WERK_DECLARE (wk, int64_t) ;
            GB_WERK_PUSH (ws, nthreads, int64_t) ;
            GB_WERK_PUSH (wk, nthreads, int64_t) ;
            if (ws == NULL || wk == NULL)
            { 
                // out of memory; use a single thread instead
                GB_WERK_POP (wk, int64_t) ;
                GB_WERK_POP (ws, int64_t) ;
                GB_cumsum (count, n, kresult, 1, NULL) ;
                return ;
            }

            int tid ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (tid = 0 ; tid < nthreads ; tid++)
            {
                // each task sums up its own part
                int64_t istart, iend ;
                GB_PARTITION (istart, iend, n, tid, nthreads) ;
                int64_t k = 0 ;
                int64_t s = 0 ;
                for (int64_t i = istart ; i < iend ; i++)
                { 
                    int64_t c = count [i] ;
                    if (c != 0) k++ ;
                    s += c ;
                }
                ws [tid] = s ;
                wk [tid] = k ;
            }

            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (tid = 0 ; tid < nthreads ; tid++)
            {
                // each task computes the cumsum of its own part
                int64_t istart, iend ;
                GB_PARTITION (istart, iend, n, tid, nthreads) ;
                int64_t s = 0 ;
                for (int i = 0 ; i < tid ; i++)
                { 
                    s += ws [i] ;
                }
                for (int64_t i = istart ; i < iend ; i++)
                { 
                    int64_t c = count [i] ;
                    count [i] = s ;
                    s += c ;
                }
                if (iend == n)
                { 
                    count [n] = s ;
                }
            }

            int64_t k = 0 ;
            for (int tid = 0 ; tid < nthreads ; tid++)
            { 
                k += wk [tid] ;
            }
            (*kresult) = k ;

            // free workspace
            GB_WERK_POP (wk, int64_t) ;
            GB_WERK_POP (ws, int64_t) ;
        }
    }
}

