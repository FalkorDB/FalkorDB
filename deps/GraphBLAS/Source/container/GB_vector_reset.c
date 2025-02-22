//------------------------------------------------------------------------------
// GB_vector_reset: empty contents of a GrB_Vector and set its length to 0
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Clears nearly all prior content of V, making V a full GrB_Vector of length
// zero.  The type is not changed.  V->user_name is preserved.  get/set
// controls (hyper_switch, bitmap_switch, [pji]_control, etc) are preserved,
// except that V->sparsity_control is revised to allow V to become a full
// vector.  All other content is freed.

#include "GB_container.h"

void GB_vector_reset
(
    GrB_Vector V
)
{
    if (V != NULL)
    { 
        GB_phybix_free ((GrB_Matrix) V) ;
        V->plen = -1 ;
        V->vlen = 0 ;
        V->vdim = 1 ;
        V->nvec = 1 ;
//      V->nvec_nonempty = 0 ;
        GB_nvec_nonempty_set ((GrB_Matrix) V, 0) ;
        V->nvals = 0 ;
        V->sparsity_control = V->sparsity_control | GxB_FULL ;
        V->is_csc = true ;
        V->p_is_32 = false ;
        V->j_is_32 = false ;
        V->i_is_32 = false ;
        V->magic = GB_MAGIC ;

        // V is now a valid GrB_Vector of length 0, in the full format
        ASSERT_VECTOR_OK (V, "V reset", GB0) ;
        ASSERT (GB_IS_FULL (V)) ;
    }
}

