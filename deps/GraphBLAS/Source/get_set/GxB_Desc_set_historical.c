//------------------------------------------------------------------------------
// GxB_Desc_set: set a field in a descriptor (historical methods)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

//------------------------------------------------------------------------------
// GxB_Desc_set_INT32:  set a descriptor option (int32_t)
//------------------------------------------------------------------------------

GrB_Info GxB_Desc_set_INT32     // set a parameter in a descriptor
(
    GrB_Descriptor desc,        // descriptor to modify
    int field,                  // parameter to change
    int32_t value               // value to change it to
)
{ 
    return (GrB_Descriptor_set_INT32 (desc, value, field)) ;
}

//------------------------------------------------------------------------------
// GxB_Desc_set_FP64: set a descriptor option (double scalar)
//------------------------------------------------------------------------------

GrB_Info GxB_Desc_set_FP64      // set a parameter in a descriptor
(
    GrB_Descriptor desc,        // descriptor to modify
    int field,                  // parameter to change
    double value                // value to change it to
)
{ 
    // no longer any settings for this method
    return (GrB_INVALID_VALUE) ;
}

//------------------------------------------------------------------------------
// GxB_Desc_set: based on va_arg
//------------------------------------------------------------------------------

GrB_Info GxB_Desc_set           // set a parameter in a descriptor
(
    GrB_Descriptor desc,        // descriptor to modify
    int field,                  // parameter to change
    ...                         // value to change it to
)
{ 
    va_list ap ;
    va_start (ap, field) ;
    int value = va_arg (ap, int) ;
    va_end (ap) ;
    return (GrB_Descriptor_set_INT32 (desc, value, field)) ;
}

