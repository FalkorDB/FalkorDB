//------------------------------------------------------------------------------
// LAGraph_SFreeContents: free the Contents returned by LAGraph_SRead.
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

void LAGraph_SFreeContents  // free the Contents returned by LAGraph_SRead
(
    // input/output
    LAGraph_Contents **Contents_handle,     // array of size ncontents
    GrB_Index ncontents
)
{
    if (Contents_handle != NULL)
    {
        LAGraph_Contents *Contents = (*Contents_handle) ;
        if (Contents != NULL)
        {
            for (GrB_Index i = 0 ; i < ncontents ; i++)
            {
                LAGraph_Free ((void **) &(Contents [i].blob), NULL) ;
            }
        }
        LAGraph_Free ((void **) Contents_handle, NULL) ;
    }
}
