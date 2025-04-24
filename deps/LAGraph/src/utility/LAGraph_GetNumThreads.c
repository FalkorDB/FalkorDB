//------------------------------------------------------------------------------
// LAGraph_GetNumThreads: get the # of threads to use
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

// LAGraph_get_nthreads: get # of threads that will be used by LAGraph.

#include "LG_internal.h"

int LAGraph_GetNumThreads
(
    // output:
    int *nthreads_outer,
    int *nthreads_inner,
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    LG_ASSERT (nthreads_outer != NULL && nthreads_inner != NULL,
        GrB_NULL_POINTER) ;

    //--------------------------------------------------------------------------
    // get number of threads
    //--------------------------------------------------------------------------

    (*nthreads_outer) = LG_nthreads_outer ;
    (*nthreads_inner) = LG_nthreads_inner ;
    return (GrB_SUCCESS) ;
}
