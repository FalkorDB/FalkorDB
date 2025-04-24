//------------------------------------------------------------------------------
// LAGraph_SFreeSet: free a set of matrices
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
#include "LAGraphX.h"

void LAGraph_SFreeSet           // free a set of matrices
(
    // input/output
    GrB_Matrix **Set_handle,    // array of GrB_Matrix of size nmatrices
    GrB_Index nmatrices         // # of matrices in the set
)
{
    if (Set_handle != NULL)
    {
        GrB_Matrix *Set = (*Set_handle) ;
        if (Set != NULL)
        {
            for (GrB_Index i = 0 ; i < nmatrices ; i++)
            {
                GrB_free (&(Set [i])) ;
            }
        }
        LAGraph_Free ((void **) Set_handle, NULL) ;
    }
}
