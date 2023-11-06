//------------------------------------------------------------------------------
// GB_AxB_saxpy_generic_template: C=A*B, C<M>=A*B, or C<!M>=A*B via saxpy method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A and B matrices have any format: hypersparse, sparse, bitmap, or full.
// C can be sparse, hypersparse, or bitmap, but not full.

#include "GB_mxm_shared_definitions.h"

{
    #if GB_GENERIC_C_IS_SPARSE_OR_HYPERSPARSE            
    {
        // C is sparse or hypersparse
        ASSERT (GB_IS_SPARSE (C) || GB_IS_HYPERSPARSE (C)) ;
        if (M == NULL)
        { 
            // C = A*B, no mask
            #define GB_NO_MASK 1
            #define GB_MASK_COMP 0
            #include "GB_AxB_saxpy3_template.c"
        }
        else if (!Mask_comp)
        { 
            // C<M> = A*B
            #define GB_NO_MASK 0
            #define GB_MASK_COMP 0
            #include "GB_AxB_saxpy3_template.c"
        }
        else
        { 
            // C<!M> = A*B
            #define GB_NO_MASK 0
            #define GB_MASK_COMP 1
            #include "GB_AxB_saxpy3_template.c"
        }
    }
    #else
    { 
        // C is bitmap
        #include "GB_AxB_saxbit_template.c"
    }
    #endif
}

