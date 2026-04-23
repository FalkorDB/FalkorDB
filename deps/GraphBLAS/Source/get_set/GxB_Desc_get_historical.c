//------------------------------------------------------------------------------
// GxB_Desc*_get: get a field in a descriptor (historical methods)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

//------------------------------------------------------------------------------
// GxB_Desc_get_INT32:  get a descriptor option (int32_t)
//------------------------------------------------------------------------------

GrB_Info GxB_Desc_get_INT32     // get a parameter from a descriptor
(
    GrB_Descriptor desc,        // descriptor to query; NULL is ok
    int field,                  // parameter to query
    int32_t *value              // return value of the descriptor
)
{ 
    return (GrB_Descriptor_get_INT32 (desc, value, field)) ;
}

//------------------------------------------------------------------------------
// GxB_Desc_get_FP64:  get a descriptor option (double)
//------------------------------------------------------------------------------

GrB_Info GxB_Desc_get_FP64      // get a parameter from a descriptor
(
    GrB_Descriptor desc,        // descriptor to query; NULL is ok
    int field,                  // parameter to query
    double *value               // return value of the descriptor
)
{ 
    // no longer any double parameters in the descriptor
    return (GrB_INVALID_VALUE) ;
}

//------------------------------------------------------------------------------
// GxB_Desc_get: based on va_arg
//------------------------------------------------------------------------------

GrB_Info GxB_Desc_get           // get a parameter from a descriptor
(
    GrB_Descriptor desc,        // descriptor to query; NULL is ok
    int field,                  // parameter to query
    ...                         // return value of the descriptor
)
{ 
    va_list ap ;
    va_start (ap, field) ;
    int32_t *value = va_arg (ap, int32_t *) ;
    va_end (ap) ;
    return (GrB_Descriptor_get_INT32 (desc, value, field)) ;
}

//------------------------------------------------------------------------------
// GxB_Descriptor_get: get a field in a descriptor (historical)
//------------------------------------------------------------------------------

GrB_Info GxB_Descriptor_get     // get a parameter from a descriptor
(
    int32_t *value,             // value of the parameter
    GrB_Descriptor desc,        // descriptor to query; NULL is ok
    int field                   // parameter to query
)
{ 
    return (GrB_Descriptor_get_INT32 (desc, value, field)) ;
}

