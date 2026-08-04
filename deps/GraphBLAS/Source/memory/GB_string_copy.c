//------------------------------------------------------------------------------
// GB_string_copy: safe string copy (alternative to strncpy)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Copy a string into a destination of known size, with a guaranteed NUL
// termination.  If the source string is too long, it is silently truncated.
// Unlike strncpy, the dest array is not padded with zeros.

#include "GB.h"

void GB_string_copy
(
    // output:
    char *dest,             // array of size dest_size
    // inputs:
    const char *source,
    size_t dest_size
)
{
    // sanity checks
    if (dest == NULL) return ;
    dest [0] = '\0' ;
    if (source == NULL) return ;

    // get the length of the string, excluding the NUL terminator
    size_t len = strlen (source) ;

    // ensure the dest does not encounter overflow
    len = GB_IMIN (len, dest_size-1) ;

    // copy the string, including the NUL terminator (if possible)
    memcpy (dest, source, len+1) ;

    // ensure the string is NUL-terminated (in the string is too long)
    dest [dest_size-1] = '\0' ;
}

