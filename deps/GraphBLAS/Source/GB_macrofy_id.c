//------------------------------------------------------------------------------
// GB_macrofy_id: string for identity/terminal value
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

const char *GB_macrofy_id // return string encoding the value
(
    // input:
    int ecode,          // enumerated identity/terminal value
    size_t zsize,       // size of value
    // output:          // (optional: either may be NULL)
    bool *has_byte,     // true if value is a single repeated byte
    uint8_t *byte       // repeated byte
)
{ 

    const char *f ;
    int has = (zsize == 1) ;    // always true if z is a single byte
    uint8_t b = 0 ;             // default byte value is zero

    switch (ecode)
    {

        //----------------------------------------------------------------------
        // for identity values and terminal values for terminal monoids
        //----------------------------------------------------------------------

        case  0 : f = "0"                   ; has = 1 ;          ; break ;
        case  1 : f = "1" ;                 ;         ; b = 0x01 ; break ;
        case  2 : f = "true"                ;         ; b = 0x01 ; break ;
        case  3 : f = "false"               ;         ;          ; break ;
        case  4 : f = "INT8_MAX"            ;         ; b = 0x7F ; break ;

        case  5 : f = "INT16_MAX"           ;         ;          ; break ;
        case  6 : f = "INT32_MAX"           ;         ;          ; break ;
        case  7 : f = "INT64_MAX"           ;         ;          ; break ;
        case  8 : f = "UINT8_MAX"           ;         ; b = 0xFF ; break ;
        case  9 : f = "UINT16_MAX"          ; has = 1 ; b = 0xFF ; break ;
        case 10 : f = "UINT32_MAX"          ; has = 1 ; b = 0xFF ; break ;
        case 11 : f = "UINT64_MAX"          ; has = 1 ; b = 0xFF ; break ;

        case 12 : f = "INFINITY"            ;         ;          ; break ;

        case 13 : f = "INT8_MIN"            ;         ; b = 0x80 ; break ;
        case 14 : f = "INT16_MIN"           ;         ;          ; break ;
        case 15 : f = "INT32_MIN"           ;         ;          ; break ;
        case 16 : f = "INT64_MIN"           ;         ;          ; break ;

        case 17 : f = "-INFINITY"           ;         ;          ; break ;

        case 18 : f = "0" /* ANY monoid */  ; has = 1 ;          ; break ;

        case 19 : f = "0xFF"                ; has = 1 ; b = 0xFF ; break ;
        case 20 : f = "0xFFFF"              ; has = 1 ; b = 0xFF ; break ;
        case 21 : f = "0xFFFFFFFF"          ; has = 1 ; b = 0xFF ; break ;
        case 22 : f = "0xFFFFFFFFFFFFFFFF"  ; has = 1 ; b = 0xFF ; break ;

        // ecodes 19 to 28 are reserved for future use

        // user-defined monoid (terminal or non-terminal) or
        // built-in non-terminal monoid
        default : f = "" ;                  ; break ;
    }

    if (has_byte != NULL) (*has_byte) = has ;
    if (byte != NULL) (*byte) = b ;
    return (f) ;
}

