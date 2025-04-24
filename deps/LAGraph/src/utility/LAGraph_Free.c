//------------------------------------------------------------------------------
// LAGraph_Free:  wrapper for free
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

// LAGraph_Free frees a block of memory obtained by LAGraph_Malloc.  It does
// nothing if p is NULL.

#include "LG_internal.h"

int LAGraph_Free            // free a block of memory and set p to NULL
(
    // input/output:
    void **p,               // pointer to object to free, does nothing if NULL
    char *msg
)
{
    LG_CLEAR_MSG ;

    if (p != NULL && (*p) != NULL)
    {
        LAGraph_Free_function (*p) ;
        (*p) = NULL ;
    }

    return (GrB_SUCCESS) ;
}
