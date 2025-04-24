//------------------------------------------------------------------------------
// LAGraph/src/test/LG_check_vector: extract contents of a vector, for testing
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

// This is slow, for testing only.
// See src/test/test_vector for the brutal test.

#include "LG_internal.h"
#include "LG_test.h"

int LG_check_vector
(
    int64_t *x,         // x (0:n-1) = X (0:n-1), of type int64_t
    GrB_Vector X,       // vector of size n
    int64_t n,
    int64_t missing     // value to assign to x(i) if X(i) is not present
)
{

    for (int64_t i = 0 ; i < n ; i++)
    {
        int64_t t ;
        int info = GrB_Vector_extractElement_INT64 (&t, X, i) ;
        x [i] = missing ;
        if (info == GrB_SUCCESS)
        {
            x [i] = t ;
        }
        else if (info != GrB_NO_VALUE)
        {
            return (info) ;
        }
    }
    return (GrB_SUCCESS) ;
}
