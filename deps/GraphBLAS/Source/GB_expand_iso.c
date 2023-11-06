//------------------------------------------------------------------------------
// GB_expand_iso: expand a scalar into an entire array
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: needed.

#include "GB.h"
#include "GB_is_nonzero.h"

void GB_expand_iso          // expand an iso scalar into an entire array
(
    void *restrict X,       // output array to expand into
    int64_t n,              // # of entries in X
    void *restrict scalar,  // scalar to expand into X
    size_t size             // size of the scalar and each entry of X
)
{

    //--------------------------------------------------------------------------
    // determine how many threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    //--------------------------------------------------------------------------
    // copy the scalar into X
    //--------------------------------------------------------------------------

    if (GB_is_nonzero (scalar, size))
    { 

        //----------------------------------------------------------------------
        // the scalar is nonzero
        //----------------------------------------------------------------------

        int64_t p ;
        int nthreads = GB_nthreads (n, chunk, nthreads_max) ;
        switch (size)
        {

            case GB_1BYTE : // bool, uint8, int8, and UDT of size 1
            {
                uint8_t a0 = (*((uint8_t *) scalar)) ;
                uint8_t *restrict Z = (uint8_t *) X ;
                #pragma omp parallel for num_threads(nthreads) schedule(static)
                for (p = 0 ; p < n ; p++)
                {
                    Z [p] = a0 ;
                }
            }
            break ;

            case GB_2BYTE : // uint16, int16, and UDT of size 2
            {
                uint16_t a0 = (*((uint16_t *) scalar)) ;
                uint16_t *restrict Z = (uint16_t *) X ;
                #pragma omp parallel for num_threads(nthreads) schedule(static)
                for (p = 0 ; p < n ; p++)
                {
                    Z [p] = a0 ;
                }
            }
            break ;

            case GB_4BYTE : // uint32, int32, float, and UDT of size 4
            {
                uint32_t a0 = (*((uint32_t *) scalar)) ;
                uint32_t *restrict Z = (uint32_t *) X ;
                #pragma omp parallel for num_threads(nthreads) schedule(static)
                for (p = 0 ; p < n ; p++)
                {
                    Z [p] = a0 ;
                }
            }
            break ;

            case GB_8BYTE : // uint64, int64, double, float complex, UDT size 8
            {
                uint64_t a0 = (*((uint64_t *) scalar)) ;
                uint64_t *restrict Z = (uint64_t *) X ;
                #pragma omp parallel for num_threads(nthreads) schedule(static)
                for (p = 0 ; p < n ; p++)
                {
                    Z [p] = a0 ;
                }
            }
            break ;

            case GB_16BYTE : // double complex, and UDT size 16
            {
                GB_blob16 a0 = (*((GB_blob16 *) scalar)) ;
                GB_blob16 *restrict Z = (GB_blob16 *) X ;
                #pragma omp parallel for num_threads(nthreads) schedule(static)
                for (p = 0 ; p < n ; p++)
                {
                    Z [p] = a0 ;
                }
            }
            break ;

            default : // user-defined types of arbitrary size
            {
                GB_void *restrict Z = (GB_void *) X ;
                #pragma omp parallel for num_threads(nthreads) schedule(static)
                for (p = 0 ; p < n ; p++)
                {
                    memcpy (Z + p*size, scalar, size) ;
                }
            }
            break ;
        }
    }
    else
    { 

        //----------------------------------------------------------------------
        // the scalar is zero: use memset
        //----------------------------------------------------------------------

        GB_memset (X, 0, n*size, nthreads_max) ;
    }
}

