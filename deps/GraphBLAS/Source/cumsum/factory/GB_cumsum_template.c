//------------------------------------------------------------------------------
// GB_cumsum_template: cumlative sum of an array (uint32_t or uint64_t)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Compute the cumulative sum of an array count[0:n], of size n+1,
// and of type uint32_t or uint64_t:

//      k = sum (count [0:n-1] != 0) ;
//      count = cumsum ([0 count[0:n-1]]) ;

// That is, count [j] on input is overwritten with sum (count [0..j-1]).
// On input, count [n] is not accessed and is implicitly zero on input.
// On output, count [n] is the total sum.

// If the type is uint32_t, integer overflow is checked.  If it occurs,
// the count array is not modified and the method returns false.

// Testing how GraphBLAS handles integer overflow would require a very large
// test problem (a matrix with over 4 billion entries).  To keep the test suite
// modest in size, an artificial integer overflow can be triggered, but only
// when GraphBLAS is compiled with test coverage, inside MATLAB
// (GraphBLAS/Tcov).

{
    #ifndef GB_NO_KRESULT
    if (kresult != NULL)
    {

        //----------------------------------------------------------------------
        // cumsum, and compute k, for uint32_t or uint64_t cases only
        //----------------------------------------------------------------------

        if (nthreads <= 2)
        {

            //------------------------------------------------------------------
            // cumsum with one thread, also compute k
            //------------------------------------------------------------------

            uint64_t s = 0 ;

            #if GB_CHECK_OVERFLOW
            { 
                for (int64_t i = 0 ; i < n ; i++)
                {
                    s += count [i] ;
                    if (s > UINT32_MAX)
                    { 
                        return (false) ;
                    }
                }
                #ifdef GBCOVER
                // pretend to fail, for test coverage only
                if (GB_Global_hack_get (5)) return (false) ;
                #endif
                s = 0 ;
            }
            #endif

            uint64_t k = 0 ;
            for (int64_t i = 0 ; i < n ; i++)
            { 
                uint64_t c = count [i] ;
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
            GB_WERK_DECLARE (ws, uint64_t) ;
            GB_WERK_DECLARE (wk, uint64_t) ;
            GB_WERK_PUSH (ws, nthreads, uint64_t) ;
            GB_WERK_PUSH (wk, nthreads, uint64_t) ;
            if (ws == NULL || wk == NULL)
            { 
                // out of memory; use a single thread instead
                GB_WERK_POP (wk, uint64_t) ;
                GB_WERK_POP (ws, uint64_t) ;
                return (GB_cumsum (count, count_is_32, n, kresult, 1, NULL)) ;
            }

            int tid ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (tid = 0 ; tid < nthreads ; tid++)
            {
                // each task sums up its own part
                int64_t istart, iend ;
                GB_PARTITION (istart, iend, n, tid, nthreads) ;
                uint64_t k = 0 ;
                uint64_t s = 0 ;
                for (int64_t i = istart ; i < iend ; i++)
                { 
                    uint64_t c = count [i] ;
                    if (c != 0) k++ ;
                    s += c ;
                }
                ws [tid] = s ;
                wk [tid] = k ;
            }

            #if GB_CHECK_OVERFLOW
            { 
                // for uint32_t case only
                uint64_t total = 0 ;
                for (tid = 0 ; tid < nthreads ; tid++)
                { 
                    total += ws [tid] ;
                }
                if (total > UINT32_MAX)
                { 
                    GB_WERK_POP (wk, uint64_t) ;
                    GB_WERK_POP (ws, uint64_t) ;
                    return (false) ;
                }
            }
            #ifdef GBCOVER
            if (GB_Global_hack_get (5))
            {
                // pretend to fail, for test coverage only
                GB_WERK_POP (wk, uint64_t) ;
                GB_WERK_POP (ws, uint64_t) ;
                return (false) ;
            }
            #endif
            #endif

            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (tid = 0 ; tid < nthreads ; tid++)
            {
                // each task computes the cumsum of its own part
                int64_t istart, iend ;
                GB_PARTITION (istart, iend, n, tid, nthreads) ;
                uint64_t s = 0 ;
                for (int i = 0 ; i < tid ; i++)
                { 
                    s += ws [i] ;
                }
                for (int64_t i = istart ; i < iend ; i++)
                { 
                    uint64_t c = count [i] ;
                    count [i] = s ;
                    s += c ;
                }
                if (iend == n)
                { 
                    count [n] = s ;
                }
            }

            uint64_t k = 0 ;
            for (int tid = 0 ; tid < nthreads ; tid++)
            { 
                k += wk [tid] ;
            }
            (*kresult) = (int64_t) k ;

            // free workspace
            GB_WERK_POP (wk, uint64_t) ;
            GB_WERK_POP (ws, uint64_t) ;
        }
    }
    else
    #endif
    {

        //----------------------------------------------------------------------
        // cumsum without k, for all types (uint32_t, uint64_t, and float)
        //----------------------------------------------------------------------

        if (nthreads <= 2)
        {

            //------------------------------------------------------------------
            // cumsum with one thread
            //------------------------------------------------------------------

            return (GB_CUMSUM1_TYPE (count, n)) ;

        }
        else
        {

            //------------------------------------------------------------------
            // cumsum with multiple threads
            //------------------------------------------------------------------

            // allocate workspace
            GB_WERK_DECLARE (ws, GB_WS_TYPE) ;
            GB_WERK_PUSH (ws, nthreads, GB_WS_TYPE) ;
            if (ws == NULL)
            { 
                // out of memory; use a single thread instead
                return (GB_CUMSUM1_TYPE (count, n)) ;
            }

            int tid ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (tid = 0 ; tid < nthreads ; tid++)
            {
                // each task sums up its own part
                int64_t istart, iend ;
                GB_PARTITION (istart, iend, n, tid, nthreads) ;
                GB_WS_TYPE s = 0 ;
                for (int64_t i = istart ; i < iend ; i++)
                { 
                    s += count [i] ;
                }
                ws [tid] = s ;
            }

            #if GB_CHECK_OVERFLOW
            { 
                // for uint32_t case only
                uint64_t total = 0 ;
                for (tid = 0 ; tid < nthreads ; tid++)
                { 
                    total += ws [tid] ;
                }
                if (total > UINT32_MAX)
                { 
                    GB_WERK_POP (ws, GB_WS_TYPE) ;
                    return (false) ;
                }
            }
            #ifdef GBCOVER
            if (GB_Global_hack_get (5))
            {
                // pretend to fail, for test coverage only
                GB_WERK_POP (ws, GB_WS_TYPE) ;
                return (false) ;
            }
            #endif
            #endif

            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (tid = 0 ; tid < nthreads ; tid++)
            {
                // each tasks computes the cumsum of its own part
                int64_t istart, iend ;
                GB_PARTITION (istart, iend, n, tid, nthreads) ;
                GB_WS_TYPE s = 0 ;
                for (int i = 0 ; i < tid ; i++)
                { 
                    s += ws [i] ;
                }
                for (int64_t i = istart ; i < iend ; i++)
                { 
                    GB_WS_TYPE c = count [i] ;
                    count [i] = s ;
                    s += c ;
                }
                if (iend == n)
                { 
                    count [n] = s ;
                }
            }

            // free workspace
            GB_WERK_POP (ws, GB_WS_TYPE) ;
        }
    }
}

#undef GB_CHECK_OVERFLOW
#undef GB_CUMSUM1_TYPE

