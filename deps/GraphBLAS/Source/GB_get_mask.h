//------------------------------------------------------------------------------
// GB_get_mask.h: get the user's mask and handle M->iso
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_GET_MASK_H
#define GB_GET_MASK_H
#include "GB_is_nonzero.h"

static inline GrB_Matrix GB_get_mask    // return M_in or NULL
(
    // inputs
    const GrB_Matrix M_in,      // user's input mask
    // input/output
    bool *Mask_comp,            // true if mask is complemented
    bool *Mask_struct           // true if mask is structural
)
{

    if (M_in != NULL && M_in->iso && !(*Mask_struct) &&
        (M_in->type->code != GB_UDT_code))
    {
        // The mask is present, iso, not structural, and does not have
        // a user-defined type.  Convert the mask to a structural mask.
        (*Mask_struct) = true ;

        // check the value of the mask; if M->x [0] is zero, then all entries
        // of M are zero.
        if (GB_is_nonzero ((GB_void *) M_in->x, M_in->type->size))
        { 
            // M->x [0] is nonzero, so M is a structural mask.
            GBURBLE ("(iso mask: struct) ") ;
            return (M_in) ;
        }
        else
        { 
            // M->x [0] is zero, so all entries present in the iso matrix M
            // have the value zero.  This is the same as a matrix M with no
            // entries at all.  If Mask_comp is true, this is the same as no
            // mask at all, with Mask_comp being false.  If Mask_comp is false,
            // then this is the same as no mask, complemented (the "quick mask"
            // case).
            GBURBLE ("(iso mask: all zero) ") ;
            (*Mask_comp) = !(*Mask_comp) ;
            return (NULL) ;
        }
    }
    else
    { 
        // no change to M_in, Mask_struct, or Mask_comp
        return (M_in) ;
    }
}

#endif

