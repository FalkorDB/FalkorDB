//------------------------------------------------------------------------------
// GrB_Descriptor_set: set a field in a descriptor (HISTORICAL)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Descriptor_set: historical method
//------------------------------------------------------------------------------

GrB_Info GrB_Descriptor_set     // set a parameter in a descriptor
(
    GrB_Descriptor desc,        // descriptor to modify
    int field,                  // parameter to change
    int value                   // value to change it to
)
{ 
    return (GrB_Descriptor_set_INT32 (desc, value, field)) ;
}

