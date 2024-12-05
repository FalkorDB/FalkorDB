//------------------------------------------------------------------------------
// GB_Mask_compatible: check input and operators for type compatibility
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// check the type and dimensions of the mask

#include "GB.h"

GrB_Info GB_Mask_compatible     // check type and dimensions of mask
(
    const GrB_Matrix M,         // mask to check
    const bool Mask_struct,     // true if M is structural
    const GrB_Matrix C,         // C<M>= ...
    const GrB_Index nrows,      // size of output if C is NULL (see GB*assign)
    const GrB_Index ncols,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check the mask M
    //--------------------------------------------------------------------------

    if (M != NULL)
    { 

        if (!Mask_struct)
        {
            // M is typecast to boolean
            if (!GB_Type_compatible (M->type, GrB_BOOL))
            { 
                GB_ERROR (GrB_DOMAIN_MISMATCH,
                    "M of type [%s] cannot be typecast to boolean",
                    M->type->name) ;
            }
            // if M is iso and can be typecasted to bool, Mask_struct has been
            // set true by GB_get_mask
            ASSERT (!M->iso) ;
        }

        // check the mask dimensions
        GrB_Index cnrows = (C == NULL) ? nrows : GB_NROWS (C) ;
        GrB_Index cncols = (C == NULL) ? ncols : GB_NCOLS (C) ;
        if (GB_NROWS (M) != cnrows || GB_NCOLS (M) != cncols)
        { 
            GB_ERROR (GrB_DIMENSION_MISMATCH,
                "M is " GBd "-by-" GBd "; "
                "does not match output dimensions (" GBu "-by-" GBu ")",
                GB_NROWS (M), GB_NCOLS (M), cnrows, cncols) ;
        }
    }

    return (GrB_SUCCESS) ;
}

