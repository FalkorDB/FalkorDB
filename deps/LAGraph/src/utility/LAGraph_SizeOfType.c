//------------------------------------------------------------------------------
// LAGraph_SizeOfType: return the sizeof(...) of a GraphBLAS GrB_Type
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

int LAGraph_SizeOfType
(
    // output:
    size_t *size,   // size of the type
    // input:
    GrB_Type type,  // GraphBLAS type
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    LG_ASSERT (type != NULL, GrB_NULL_POINTER) ;
    LG_ASSERT (size != NULL, GrB_NULL_POINTER) ;
    (*size) = 0 ;

    //--------------------------------------------------------------------------
    // determine the size of the type
    //--------------------------------------------------------------------------

    #if LAGRAPH_SUITESPARSE

        // always succeeds, even for user-defined types, unless the
        // type is an invalid object
        return (GxB_Type_size (size, type)) ;

    #else

        if      (type == GrB_BOOL  ) (*size) = sizeof (bool)     ;
        else if (type == GrB_INT8  ) (*size) = sizeof (int8_t)   ;
        else if (type == GrB_INT16 ) (*size) = sizeof (int16_t)  ;
        else if (type == GrB_INT32 ) (*size) = sizeof (int32_t)  ;
        else if (type == GrB_INT64 ) (*size) = sizeof (int64_t)  ;
        else if (type == GrB_UINT8 ) (*size) = sizeof (uint8_t)  ;
        else if (type == GrB_UINT16) (*size) = sizeof (uint16_t) ;
        else if (type == GrB_UINT32) (*size) = sizeof (uint32_t) ;
        else if (type == GrB_UINT64) (*size) = sizeof (uint64_t) ;
        else if (type == GrB_FP32  ) (*size) = sizeof (float)    ;
        else if (type == GrB_FP64  ) (*size) = sizeof (double)   ;
        #if 0
        else if (type == GxB_FC32  ) (*size) = sizeof (GxB_FC32_t) ;
        else if (type == GxB_FC64  ) (*size) = sizeof (GxB_FC64_t) ;
        #endif
        else
        {
            LG_ASSERT_MSG (false,
                GrB_NOT_IMPLEMENTED, "user-defined types not supported") ;
        }
        return (GrB_SUCCESS) ;

    #endif
}
