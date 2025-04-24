//------------------------------------------------------------------------------
// LAGraph_Cached_EMin: determine G->emin
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

#define LG_FREE_ALL             \
{                               \
    GrB_free (&G->emin) ;       \
}

#include "LG_internal.h"

int LAGraph_Cached_EMin
(
    // input/output:
    LAGraph_Graph G,    // graph to determine G->emin
    char *msg
)
{

    //--------------------------------------------------------------------------
    // clear msg and check G
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG_AND_BASIC_ASSERT (G, msg) ;

    if (G->emin != NULL)
    {
        // G->emin already computed
        return (GrB_SUCCESS) ;
    }

    G->emin_state = LAGRAPH_UNKNOWN ;

    //--------------------------------------------------------------------------
    // determine the type of G->A and the corresponding min monoid
    //--------------------------------------------------------------------------

    char atype_name [LAGRAPH_MAX_NAME_LEN] ;
    LG_TRY (LAGraph_Matrix_TypeName (atype_name, G->A, msg)) ;
    GrB_Type atype ;
    LG_TRY (LAGraph_TypeFromName (&atype, atype_name, msg)) ;
    GrB_Monoid monoid ;
    if      (atype == GrB_BOOL  ) monoid = GrB_LAND_MONOID_BOOL  ;
    else if (atype == GrB_INT8  ) monoid = GrB_MIN_MONOID_INT8   ;
    else if (atype == GrB_INT16 ) monoid = GrB_MIN_MONOID_INT16  ;
    else if (atype == GrB_INT32 ) monoid = GrB_MIN_MONOID_INT32  ;
    else if (atype == GrB_INT64 ) monoid = GrB_MIN_MONOID_INT64  ;
    else if (atype == GrB_UINT8 ) monoid = GrB_MIN_MONOID_UINT8  ;
    else if (atype == GrB_UINT16) monoid = GrB_MIN_MONOID_UINT16 ;
    else if (atype == GrB_UINT32) monoid = GrB_MIN_MONOID_UINT32 ;
    else if (atype == GrB_UINT64) monoid = GrB_MIN_MONOID_UINT64 ;
    else if (atype == GrB_FP32  ) monoid = GrB_MIN_MONOID_FP32   ;
    else if (atype == GrB_FP64  ) monoid = GrB_MIN_MONOID_FP64   ;
    else
    {
        LG_ASSERT_MSG (false, GrB_NOT_IMPLEMENTED, "type not supported") ;
    }

    //--------------------------------------------------------------------------
    // compute G->emin
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Scalar_new (&(G->emin), atype)) ;
    GRB_TRY (GrB_reduce (G->emin, NULL, monoid, G->A, NULL)) ;
    G->emin_state = LAGraph_VALUE ;
    return (GrB_SUCCESS) ;
}
