//------------------------------------------------------------------------------
// LG_brutal_teardown.c: teardown an LAGraph test with brutal memory testing
// -----------------------------------------------------------------------------

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
#include "LG_test.h"

int LG_brutal_teardown (char *msg)
{
    LG_TRY (LAGraph_Finalize (msg)) ;
    // nothing must be left allocated
    if (LG_nmalloc != 0) printf ("Leak! %g\n", (double) LG_nmalloc) ;
    return ((LG_nmalloc == 0) ? 0 : -911) ;
}
