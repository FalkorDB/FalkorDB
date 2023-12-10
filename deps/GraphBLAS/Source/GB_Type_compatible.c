//------------------------------------------------------------------------------
// GB_Type_compatible: return true if domains are compatible
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Two domains are compatible for typecasting between them if both are built-in
// types (of any kind) or if both are the same user-defined type.

#include "GB.h"

bool GB_Type_compatible             // check if two types can be typecast
(
    const GrB_Type atype,
    const GrB_Type btype
)
{
    if (atype == NULL || btype == NULL)
    { 
        // the op ignores its inputs
        return (true) ;
    }
    else if (atype->code == GB_UDT_code || btype->code == GB_UDT_code)
    { 
        // two user types must be identical to be compatible
        return (atype == btype) ;
    }
    else
    { 
        // any built-in domain is compatible with any other built-in domain
        return (true) ;
    }
}

