//------------------------------------------------------------------------------
// LAGraph_TypeFromName: return the GrB_Type corresponding to its given name
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

// This method works for any GraphBLAS library.  On input, name is a char array
// of length at least LAGRAPH_MAX_NAME_LEN.

// Only built-in types are supported.  User-defined types are not supported.

#include "LG_internal.h"

int LAGraph_TypeFromName
(
    // output:
    GrB_Type *type, // GraphBLAS type
    // input:
    char *name,     // name of the type: a null-terminated string
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    LG_ASSERT (type != NULL, GrB_NULL_POINTER) ;
    LG_ASSERT (name != NULL, GrB_NULL_POINTER) ;

    //--------------------------------------------------------------------------
    // determine the GrB_Type from its name
    //--------------------------------------------------------------------------

    #if LAGRAPH_SUITESPARSE

        return (GxB_Type_from_name (type, name)) ;

    #else

        if      (MATCHNAME (name, "bool"          )) (*type) = GrB_BOOL   ;
        else if (MATCHNAME (name, "int8_t"        )) (*type) = GrB_INT8   ;
        else if (MATCHNAME (name, "int16_t"       )) (*type) = GrB_INT16  ;
        else if (MATCHNAME (name, "int32_t"       )) (*type) = GrB_INT32  ;
        else if (MATCHNAME (name, "int64_t"       )) (*type) = GrB_INT64  ;
        else if (MATCHNAME (name, "uint8_t"       )) (*type) = GrB_UINT8  ;
        else if (MATCHNAME (name, "uint16_t"      )) (*type) = GrB_UINT16 ;
        else if (MATCHNAME (name, "uint32_t"      )) (*type) = GrB_UINT32 ;
        else if (MATCHNAME (name, "uint64_t"      )) (*type) = GrB_UINT64 ;
        else if (MATCHNAME (name, "float"         )) (*type) = GrB_FP32   ;
        else if (MATCHNAME (name, "double"        )) (*type) = GrB_FP64   ;
        #if 0
        // if complex types from SuiteSparse:GraphBLAS are added to LAGraph:
        else if (MATCHNAME (name, "float complex" )) (*type) = GxB_FC32   ;
        else if (MATCHNAME (name, "double complex")) (*type) = GxB_FC64   ;
        #endif
        else
        {
            (*type) = NULL ;
            LG_ASSERT_MSGF (false, GrB_NOT_IMPLEMENTED,
                "type \"%s\" not supported", name) ;
        }
        return (GrB_SUCCESS) ;

    #endif
}
