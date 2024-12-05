//------------------------------------------------------------------------------
// GB_enumify_masker: enumerate a masker problem
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

uint64_t GB_enumify_masker  // enumify a masker problem
(
    // output:
    uint64_t *method_code,  // unique encoding of the entire operation
    // input:
    const GrB_Matrix R,     // NULL for phase 1
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Matrix C,
    const GrB_Matrix Z
)
{ 

    //--------------------------------------------------------------------------
    // get the types of R, C, and Z
    //--------------------------------------------------------------------------

    GrB_Type rtype = (R == NULL) ? NULL : R->type ;
    ASSERT (GB_IMPLIES (R != NULL, rtype == C->type)) ;
    ASSERT (GB_IMPLIES (R != NULL, rtype == Z->type)) ;

    //--------------------------------------------------------------------------
    // enumify the types
    //--------------------------------------------------------------------------

    int rcode = (rtype == NULL) ? 0 : rtype->code ;     // 0 to 14
    int C_iso_code = (C->iso || rtype == NULL) ? 1 : 0 ;
    int Z_iso_code = (Z->iso || rtype == NULL) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // enumify the mask
    //--------------------------------------------------------------------------

    int mtype_code = M->type->code ; // 0 to 14
    int mask_ecode ;
    GB_enumify_mask (&mask_ecode, mtype_code, Mask_struct, Mask_comp) ;

    //--------------------------------------------------------------------------
    // enumify the sparsity structures of R, M, C, and Z
    //--------------------------------------------------------------------------

    int R_sparsity = GB_sparsity (R) ;
    int M_sparsity = GB_sparsity (M) ;
    int C_sparsity = GB_sparsity (C) ;
    int Z_sparsity = GB_sparsity (Z) ;

    int rsparsity, msparsity, csparsity, zsparsity ;
    GB_enumify_sparsity (&rsparsity, R_sparsity) ;
    GB_enumify_sparsity (&msparsity, M_sparsity) ;
    GB_enumify_sparsity (&csparsity, C_sparsity) ;
    GB_enumify_sparsity (&zsparsity, Z_sparsity) ;

    //--------------------------------------------------------------------------
    // construct the masker method_code
    //--------------------------------------------------------------------------

    // total method_code bits: 18 (5 hex digits)

    (*method_code) =
                                               // range        bits
                // C and Z iso properites (1 hex digit)
                GB_LSHIFT (C_iso_code , 17) |  // 0 or 1       1
                GB_LSHIFT (Z_iso_code , 16) |  // 0 or 1       1

                // mask (1 hex digit)
                GB_LSHIFT (mask_ecode , 12) |  // 0 to 13      4

                // type of R (1 hex digit)
                GB_LSHIFT (rcode      ,  8) |  // 0 to 14      4

                // sparsity structures of R, M, C, and Z (2 hex digits)
                GB_LSHIFT (rsparsity  ,  6) |  // 0 to 3       2
                GB_LSHIFT (msparsity  ,  4) |  // 0 to 3       2
                GB_LSHIFT (csparsity  ,  2) |  // 0 to 3       2
                GB_LSHIFT (zsparsity  ,  0) ;  // 0 to 3       2

}

