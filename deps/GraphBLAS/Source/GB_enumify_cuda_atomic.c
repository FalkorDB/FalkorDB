//------------------------------------------------------------------------------
// GB_enumify_cuda_atomic: enumify the CUDA atomic for a monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

bool GB_enumify_cuda_atomic
(
    // output:
    const char **a,                 // CUDA atomic function name
    bool *user_monoid_atomically,   // true if user monoid has an atomic update
    const char **cuda_type,         // CUDA atomic type
    // input:
    GrB_Monoid monoid,  // monoid to query
    int add_ecode,      // binary op as an enum
    size_t zsize,       // ztype->size
    int zcode           // ztype->code
)
{ 

    // All built-in monoids are handled, except for the double complex cases of
    // ANY and TIMES.  Those need to be done the same way user-defined monoids
    // are computed.

    (*a) = NULL ;
    (*user_monoid_atomically) = false ;
    (*cuda_type) = NULL ;
    bool has_cheeseburger = false ;

    switch (add_ecode)
    {

        // user defined monoid: can apply GB_ADD via atomicCAS if the ztype has
        // 16, 32, or 64 bits
        case  0 :

            (*user_monoid_atomically) =
                (zsize == sizeof (uint16_t) ||
                 zsize == sizeof (uint32_t) ||
                 zsize == sizeof (uint64_t))  ;
            break ;

        // FIRST, ANY, SECOND: atomic write (not double complex)
        case  1 :
        case  2 :

            switch (zcode)
            {
                case GB_BOOL_code    : 
                case GB_INT8_code    : 
                case GB_UINT8_code   : 
                case GB_INT16_code   : 
                case GB_UINT16_code  : 
                case GB_INT32_code   : 
                case GB_UINT32_code  : 
                case GB_INT64_code   : 
                case GB_UINT64_code  : 
                case GB_FP32_code    : 
                case GB_FP64_code    : 
                case GB_FC32_code    : (*a) = "GB_cuda_atomic_write" ;
                default              : break ;
            }
            break ;

        // MIN (real only)
        case  3 :
        case  4 :
        case  5 :

            switch (zcode)
            {
                case GB_INT8_code    : 
                case GB_UINT8_code   : 
                case GB_INT16_code   : 
                case GB_UINT16_code  : 
                case GB_INT32_code   : 
                case GB_UINT32_code  : 
                case GB_INT64_code   : 
                case GB_UINT64_code  : 
                case GB_FP32_code    : 
                case GB_FP64_code    : (*a) = "GB_cuda_atomic_min" ;
                default              : break ;
            }
            break ;

        // MAX (real only)
        case  6 :
        case  7 :
        case  8 :

            switch (zcode)
            {
                case GB_INT8_code    : 
                case GB_UINT8_code   : 
                case GB_INT16_code   : 
                case GB_UINT16_code  : 
                case GB_INT32_code   : 
                case GB_UINT32_code  : 
                case GB_INT64_code   : 
                case GB_UINT64_code  : 
                case GB_FP32_code    : 
                case GB_FP64_code    : (*a) = "GB_cuda_atomic_max" ;
                default              : break ;
            }
            break ;

        // PLUS:  all types
        case  9 :
        case 10 :
        case 11 :

            switch (zcode)
            {
                case GB_INT8_code    : 
                case GB_UINT8_code   : 
                case GB_INT16_code   : 
                case GB_UINT16_code  : 
                case GB_INT32_code   : 
                case GB_UINT32_code  : 
                case GB_INT64_code   : 
                case GB_UINT64_code  : 
                case GB_FP32_code    : 
                case GB_FP64_code    : 
                case GB_FC32_code    : 
                case GB_FC64_code    : (*a) = "GB_cuda_atomic_add" ;
                default              : break ;
            }
            break ;

        // TIMES: all real types, and float complex (but not double complex)
        case 12 : 
        case 14 : 

            switch (zcode)
            {
                case GB_INT8_code    : 
                case GB_UINT8_code   : 
                case GB_INT16_code   : 
                case GB_UINT16_code  : 
                case GB_INT32_code   : 
                case GB_UINT32_code  : 
                case GB_INT64_code   : 
                case GB_UINT64_code  : 
                case GB_FP32_code    : 
                case GB_FP64_code    : 
                case GB_FC32_code    : (*a) = "GB_cuda_atomic_times" ;
                default              : break ;
            }
            break ;

        // BOR: z = (x | y), bitwise or,
        // logical LOR (via upscale to uint32_t and BOR)
        case 17 :
        case 19 :

            switch (zcode)
            {
                case GB_BOOL_code    : 
                case GB_UINT8_code   : 
                case GB_UINT16_code  : 
                case GB_UINT32_code  : 
                case GB_UINT64_code  : (*a) = "GB_cuda_atomic_bor" ;
                default              : break ;
            }
            break ;

        // BAND: z = (x & y), bitwise and
        // logical LAND (via upscale to uint32_t and BAND)
        case 18 :
        case 20 :

            switch (zcode)
            {
                case GB_BOOL_code    : 
                case GB_UINT8_code   : 
                case GB_UINT16_code  : 
                case GB_UINT32_code  : 
                case GB_UINT64_code  : (*a) = "GB_cuda_atomic_band" ;
                default              : break ;
            }
            break ;

        // BXOR: z = (x ^ y), bitwise xor, and boolean LXOR
        case 16 :
        case 21 :

            switch (zcode)
            {
                case GB_BOOL_code    : 
                case GB_UINT8_code   : 
                case GB_UINT16_code  : 
                case GB_UINT32_code  : 
                case GB_UINT64_code  : (*a) = "GB_cuda_atomic_bxor" ;
                default              : break ;
            }
            break ;

        // BXNOR: z = ~(x ^ y), bitwise xnor, and boolean LXNOR
        case 15 :
        case 22 :

            switch (zcode)
            {
                case GB_BOOL_code    : 
                case GB_UINT8_code   : 
                case GB_UINT16_code  : 
                case GB_UINT32_code  : 
                case GB_UINT64_code  : (*a) = "GB_cuda_atomic_bxnor" ;
                default              : break ;
            }
            break ;

        // all other monoids
        default: break ;
    }

    if (monoid == NULL || zcode == 0)
    { 

        //----------------------------------------------------------------------
        // C is iso: no values computed so no need for any CUDA atomics
        //----------------------------------------------------------------------

        has_cheeseburger = false ;

    }
    else if (*user_monoid_atomically)
    { 

        //----------------------------------------------------------------------
        // user-defined monoid with a type of 16, 32, or 64 bits
        //----------------------------------------------------------------------

        if (zsize == sizeof (uint16_t))
        {
            (*cuda_type) = "unsigned short int" ;
        }
        else if (zsize == sizeof (uint32_t))
        {
            (*cuda_type) = "unsigned int" ;
        }
        else // if (zsize == sizeof (uint64_t))
        {
            (*cuda_type) = "unsigned long long int" ;
        }
        (*a) = "GB_cuda_atomic_user" ;
        has_cheeseburger = true ;

    }
    else if ((*a) == NULL)
    { 

        //----------------------------------------------------------------------
        // no CUDA atomic available
        //----------------------------------------------------------------------

        // either built-in (GxB_ANY_FC64_MONOID or GxB_TIMES_FC64_MONOID),
        // or user-defined where the type is not 16, 32, or 64 bits in size

        has_cheeseburger = false ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // CUDA atomic available for a built-in monoid
        //----------------------------------------------------------------------

        // upscale 8-bit and 16-bit types to 32-bits,
        // all others use their native types
        switch (zcode)
        {
            case GB_INT8_code    : 
            case GB_INT16_code   : 
            case GB_INT32_code   : (*cuda_type) = "int32_t"    ; break ;
            case GB_INT64_code   : (*cuda_type) = "int64_t"    ; break ;
            case GB_BOOL_code    : 
            case GB_UINT8_code   : 
            case GB_UINT16_code  : 
            case GB_UINT32_code  : (*cuda_type) = "uint32_t"   ; break ;
            case GB_UINT64_code  : (*cuda_type) = "uint64_t"   ; break ;
            case GB_FP32_code    : (*cuda_type) = "float"      ; break ;
            case GB_FP64_code    : (*cuda_type) = "double"     ; break ;
            case GB_FC32_code    : (*cuda_type) = "GxB_FC32_t" ; break ;
            case GB_FC64_code    : (*cuda_type) = "GxB_FC64_t" ; break ;
            default :;
        }
        has_cheeseburger = true ;
    }

    return (has_cheeseburger) ;
}

