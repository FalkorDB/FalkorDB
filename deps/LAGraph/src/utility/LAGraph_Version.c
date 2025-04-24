//------------------------------------------------------------------------------
// LAGraph_Version: return the LAGraph version number and date
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

// The version number and date can also be obtained via compile-time constants
// from LAGraph.h.  However, it is possible to compile a user application that
// #includes one version of LAGraph.h and then links with another version of
// the LAGraph library later on, so the version number and date may differ from
// the compile-time constants.

// The LAGraph_Version method allows the library itself to be
// queried, after it is linked in with the user application.

// The version_number array is set to LAGRAPH_VERSION_MAJOR,
// LAGRAPH_VERSION_MINOR, and LAGRAPH_VERSION_UPDATE, in that order.
// The LAGRAPH_DATE string is copied into the user-provided version_date
// string, and is null-terminated.

#include "LG_internal.h"

int LAGraph_Version
(
    // output:
    int version_number [3],     // user-provided array of size 3
    char version_date [LAGRAPH_MSG_LEN],    // user-provided array
    char *msg
)
{

    LG_CLEAR_MSG ;

    // check inputs
    LG_ASSERT (version_number != NULL && version_date != NULL,
        GrB_NULL_POINTER) ;

    // get version number and date
    version_number [0] = LAGRAPH_VERSION_MAJOR ;
    version_number [1] = LAGRAPH_VERSION_MINOR ;
    version_number [2] = LAGRAPH_VERSION_UPDATE ;
    strncpy (version_date, LAGRAPH_DATE, LAGRAPH_MSG_LEN) ;

    // make sure the date is null-terminated
    version_date [LAGRAPH_MSG_LEN-1] = '\0' ;

    return (GrB_SUCCESS) ;
}
