//------------------------------------------------------------------------------
// GB_serialize_method: parse the compression method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_serialize.h"

void GB_serialize_method
(
    // output
    int32_t *algo,                  // algorithm to use
    int32_t *level,                 // compression level
    // input
    int32_t method
)
{

    if (method < 0)
    { 
        // no compression if method is negative
        (*algo) = GxB_COMPRESSION_NONE ;
        (*level) = 0 ;
        return ;
    }

    // Determine the algorithm and level.  Lower levels give faster compression
    // time but not as good of compression.  Higher levels give more compact
    // compressions, at the cost of higher run times.  For all methods: a level
    // of zero, or a level setting outside the range permitted for a method,
    // means that default level for that method is used.
    (*algo) = 1000 * (method / 1000) ;
    (*level) = method % 1000 ;

    switch (*algo)
    {

        default : 
            // The default method has changed to ZSTD, level 1, as of
            // SuiteSparse:GraphBLAS v7.2.0.
            (*algo) = GxB_COMPRESSION_ZSTD ; 
            (*level) = 1 ;              // fast with good compression
            break ;

        case GxB_COMPRESSION_LZ4 : 
            (*level) = 0 ;              // level is ignored
            break ;

        case GxB_COMPRESSION_LZ4HC :    // LZ4HC: level 1 to 9; default 9.
            // Note that LZ4HC supports levels 10, 11, and 12, but these are
            // very slow and do not provide much benefit over level 9.  Level
            // 10 often results in a larger blob than level 9.  Level 12 is
            // typically just a tiny bit more compact than level 9, but can be
            // 10x slower, or worse, as compared to level 9.
            if ((*level) <= 0 || (*level) > 9) (*level) = 9 ;
            break ;

        case GxB_COMPRESSION_ZSTD :     // ZSTD: level 1 to 19; default 1.
            if ((*level) <= 0 || (*level) > 19) (*level) = 1 ;
            break ;
    }
}

