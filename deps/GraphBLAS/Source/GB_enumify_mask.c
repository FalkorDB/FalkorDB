//------------------------------------------------------------------------------
// GB_enumify_mask: return mask_ecode to define mask macros
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// User-defined types cannot be used as a valued mask.  They can be used
// as structural masks.

#include "GB.h"
#include "GB_stringify.h"

void GB_enumify_mask       // return enum to define mask macros
(
    // output:
    int *mask_ecode,            // enumified mask
    // input
    const GB_Type_code mcode,   // typecode of the mask matrix M,
                                // or 0 if M is not present
    bool Mask_struct,           // true if M structural, false if valued
    bool Mask_comp              // true if M complemented
)
{

    // Mask_comp becomes the least significant bit, so that
    // Mask_comp = (mask_ecode & 0x1) can be computed later.
    // Mask_struct = (mask_ecode == 2 || mask_ecode == 3)

    int e = -1 ;

    if (mcode == 0)
    { 

        //----------------------------------------------------------------------
        // no mask is present
        //----------------------------------------------------------------------

        // Mask_struct is ignored, but the mask may be complemented or not

        // user-defined types are OK.  The values of M are not accessed.
        e = (!Mask_comp) ? 0 : 1 ;

    }
    else if (Mask_struct)
    { 

        //----------------------------------------------------------------------
        // M is present, and structural (not valued).
        //----------------------------------------------------------------------

        // user-defined types are OK.  The values of M are not accessed.
        e = (!Mask_comp) ? 2 : 3 ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // M is present, and valued
        //----------------------------------------------------------------------

        switch (mcode)
        {

            case GB_BOOL_code  : 
            case GB_INT8_code  : 
            case GB_UINT8_code : 

                // valued mask, values are 1 byte in size
                e = (!Mask_comp) ? 4 : 5 ;
                break ;

            case GB_INT16_code  : 
            case GB_UINT16_code : 

                // valued mask, values are 2 bytes in size
                e = (!Mask_comp) ? 6 : 7 ;
                break ;

            case GB_INT32_code  : 
            case GB_UINT32_code : 
            case GB_FP32_code   : 

                // valued mask, values are 4 bytes in size
                e = (!Mask_comp) ? 8 : 9 ;
                break ;

            case GB_INT64_code  : 
            case GB_UINT64_code : 
            case GB_FC32_code   : 
            case GB_FP64_code   : 

                // valued mask, values are 8 bytes in size
                e = (!Mask_comp) ? 10 : 11 ;
                break ;

            case GB_FC64_code   : 

                // valued mask, values are 16 bytes in size
                e = (!Mask_comp) ? 12 : 13 ;
                break ;

            default: ;
                // user-defined types are not allowed
                e = -1 ;
                break ;
        }
    }

    (*mask_ecode) = e ;
}

