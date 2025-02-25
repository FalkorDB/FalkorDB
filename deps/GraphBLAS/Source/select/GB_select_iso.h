//------------------------------------------------------------------------------
// GB_select_iso.h: copy the iso value into C for select methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_SELECT_ISO_H
#define GB_SELECT_ISO_H

//------------------------------------------------------------------------------
// GB_select_iso: assign the iso value of C for GB_*selector
//------------------------------------------------------------------------------

static inline void GB_select_iso
(
    GB_void *Cx,                    // output iso value (same type as A)
    const GB_Opcode opcode,         // selector opcode
    const GB_void *athunk,          // thunk scalar, of size asize
    const GB_void *Ax,              // Ax [0] scalar, of size asize
    const size_t asize
)
{
    if (opcode == GB_VALUEEQ_idxunop_code)
    { 
        // all entries in C are equal to thunk
        memcpy (Cx, athunk, asize) ;
    }
    else
    { 
        // A and C are both iso
        memcpy (Cx, Ax, asize) ;
    }
}

#endif

