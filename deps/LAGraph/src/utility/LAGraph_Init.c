//------------------------------------------------------------------------------
// LAGraph_Init: start GraphBLAS and LAGraph
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

#include "LG_internal.h"

int LAGraph_Init (char *msg)
{

    LG_CLEAR_MSG ;

    // use GraphBLAS nonblocking mode, and ANSI C memory allocation functions
    return (LAGr_Init (GrB_NONBLOCKING, malloc, calloc, realloc, free, msg)) ;
}
