//------------------------------------------------------------------------------
// GB_cast_int: parallel memcpy or int32_t/int64_t/uint32_t/uint64_t type cast
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

void GB_cast_int                // parallel memcpy/cast of integer arrays
(
    void *dest,                 // destination
    GB_Type_code dest_code,     // destination type: int32/64, or uint32/64
    const void *src,            // source
    GB_Type_code src_code,      // source type: int32/64, or uint32/64
    size_t n,                   // # of entries to copy
    int nthreads_max            // max # of threads to use
)
{

    //--------------------------------------------------------------------------
    // get the # of threads
    //--------------------------------------------------------------------------

    int nthreads = GB_nthreads (n, GB_CHUNK_DEFAULT, nthreads_max) ;
    int64_t k ;

    //--------------------------------------------------------------------------
    // copy/cast the integer array
    //--------------------------------------------------------------------------

    switch (dest_code)
    {

        //----------------------------------------------------------------------
        // destination is int32_t
        //----------------------------------------------------------------------

        case GB_INT32_code :

            switch (src_code)
            {
                case GB_INT32_code :
                case GB_UINT32_code : 
                    GB_memcpy (dest, src, n * sizeof (uint32_t), nthreads) ;
                    break ;

                case GB_INT64_code :
                {
                    int32_t *restrict Dest = (int32_t *) dest ;
                    const int64_t *restrict Src = (int64_t *) src ;
                    #include "cast/factory/GB_cast_int_template.c"
                }
                break ;

                case GB_UINT64_code :
                {
                    int32_t *restrict Dest = (int32_t *) dest ;
                    const uint64_t *restrict Src = (uint64_t *) src ;
                    #include "cast/factory/GB_cast_int_template.c"
                }
                break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        // destination is uint32_t
        //----------------------------------------------------------------------

        case GB_UINT32_code :

            switch (src_code)
            {
                case GB_INT32_code :
                case GB_UINT32_code : 
                    GB_memcpy (dest, src, n * sizeof (uint32_t), nthreads) ;
                    break ;

                case GB_INT64_code :
                {
                    uint32_t *restrict Dest = (uint32_t *) dest ;
                    const int64_t *restrict Src = (int64_t *) src ;
                    #include "cast/factory/GB_cast_int_template.c"
                }
                break ;

                case GB_UINT64_code :
                {
                    uint32_t *restrict Dest = (uint32_t *) dest ;
                    const uint64_t *restrict Src = (uint64_t *) src ;
                    #include "cast/factory/GB_cast_int_template.c"
                }
                break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        // destination is int64_t
        //----------------------------------------------------------------------

        case GB_INT64_code :

            switch (src_code)
            {
                case GB_INT32_code :
                {
                    int64_t *restrict Dest = (int64_t *) dest ;
                    const int32_t *restrict Src = (int32_t *) src ;
                    #include "cast/factory/GB_cast_int_template.c"
                }
                break ;

                case GB_UINT32_code : 
                {
                    int64_t *restrict Dest = (int64_t *) dest ;
                    const uint32_t *restrict Src = (uint32_t *) src ;
                    #include "cast/factory/GB_cast_int_template.c"
                }
                break ;

                case GB_INT64_code :
                case GB_UINT64_code :
                    GB_memcpy (dest, src, n * sizeof (uint64_t), nthreads) ;
                    break ;

                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        // destination is uint64_t
        //----------------------------------------------------------------------

        case GB_UINT64_code :

            switch (src_code)
            {
                case GB_INT32_code :
                {
                    uint64_t *restrict Dest = (uint64_t *) dest ;
                    const int32_t *restrict Src = (int32_t *) src ;
                    #include "cast/factory/GB_cast_int_template.c"
                }
                break ;

                case GB_UINT32_code : 
                {
                    uint64_t *restrict Dest = (uint64_t *) dest ;
                    const uint32_t *restrict Src = (uint32_t *) src ;
                    #include "cast/factory/GB_cast_int_template.c"
                }
                break ;

                case GB_INT64_code :
                case GB_UINT64_code :
                    GB_memcpy (dest, src, n * sizeof (uint64_t), nthreads) ;
                    break ;

                default: ;
            }
            break ;

        default: ;
    }
}

