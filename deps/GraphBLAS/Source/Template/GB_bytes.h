//------------------------------------------------------------------------------
// GB_bytes.h: sizes of built-in types
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// These definitions are normally 1, 2, 4, 8, and 16, but using sizeof (...)
// to ensure portability.

#ifndef GB_BYTES_H
#define GB_BYTES_H

#define GB_1BYTE  (sizeof (uint8_t))
#define GB_2BYTE  (sizeof (uint16_t))
#define GB_4BYTE  (sizeof (uint32_t))
#define GB_8BYTE  (sizeof (uint64_t))
#define GB_16BYTE (sizeof (GB_blob16))

typedef struct
{
    uint64_t stuff [2] ;            // not accessed directly
}
GB_blob16 ;                         // sizeof (GB_blob16) is GB_16BYTE

#endif

