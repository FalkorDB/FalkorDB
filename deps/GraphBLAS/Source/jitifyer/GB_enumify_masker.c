//------------------------------------------------------------------------------
// GB_enumify_masker: enumerate a masker problem
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_enumify_masker      // enumify a masker problem
(
    // output:
    uint64_t *method_code,  // unique encoding of the entire operation
    // input:
    const GrB_Matrix R,     // NULL for phase 1
    const bool Rp_is_32,    // if true, R->p is 32-bit; else 64-bit
    const bool Rj_is_32,    // if true, R->h is 32-bit; else 64-bit
    const bool Ri_is_32,    // if true, R->i is 32-bit; else 64-bit
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
    // enumify the sparsity structures of R, C, M, and Z
    //--------------------------------------------------------------------------

    int R_sparsity = GB_sparsity (R) ;
    int C_sparsity = GB_sparsity (C) ;
    int M_sparsity = GB_sparsity (M) ;
    int Z_sparsity = GB_sparsity (Z) ;

    int rsparsity, csparsity, msparsity, zsparsity ;
    GB_enumify_sparsity (&rsparsity, R_sparsity) ;
    GB_enumify_sparsity (&csparsity, C_sparsity) ;
    GB_enumify_sparsity (&msparsity, M_sparsity) ;
    GB_enumify_sparsity (&zsparsity, Z_sparsity) ;

    int rp_is_32 = (Rp_is_32  ) ? 1 : 0 ;
    int rj_is_32 = (Rj_is_32  ) ? 1 : 0 ;
    int ri_is_32 = (Ri_is_32  ) ? 1 : 0 ;

    int cp_is_32 = (C->p_is_32) ? 1 : 0 ;
    int cj_is_32 = (C->j_is_32) ? 1 : 0 ;
    int ci_is_32 = (C->i_is_32) ? 1 : 0 ;

    int mp_is_32 = (M->p_is_32) ? 1 : 0 ;
    int mj_is_32 = (M->j_is_32) ? 1 : 0 ;
    int mi_is_32 = (M->i_is_32) ? 1 : 0 ;

    int zp_is_32 = (Z->p_is_32) ? 1 : 0 ;
    int zj_is_32 = (Z->j_is_32) ? 1 : 0 ;
    int zi_is_32 = (Z->i_is_32) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // construct the masker method_code
    //--------------------------------------------------------------------------

    // total method_code bits: 30 (8 hex digits)

    (*method_code) =
                                               // range        bits

                // R, C, M, Z: 32/64 bits (12 bits, 3 hex digits)
                GB_LSHIFT (rp_is_32   , 31) |  // 0 or 1       1
                GB_LSHIFT (rj_is_32   , 30) |  // 0 or 1       1
                GB_LSHIFT (ri_is_32   , 29) |  // 0 or 1       1

                GB_LSHIFT (cp_is_32   , 28) |  // 0 or 1       1
                GB_LSHIFT (cj_is_32   , 27) |  // 0 or 1       1
                GB_LSHIFT (ci_is_32   , 26) |  // 0 or 1       1

                GB_LSHIFT (mp_is_32   , 25) |  // 0 or 1       1
                GB_LSHIFT (mj_is_32   , 24) |  // 0 or 1       1
                GB_LSHIFT (mi_is_32   , 23) |  // 0 or 1       1

                GB_LSHIFT (zp_is_32   , 22) |  // 0 or 1       1
                GB_LSHIFT (zj_is_32   , 21) |  // 0 or 1       1
                GB_LSHIFT (zi_is_32   , 20) |  // 0 or 1       1

                // C and Z iso properites (1 hex digit)
                // unused: 2 bits
                GB_LSHIFT (C_iso_code , 17) |  // 0 or 1       1
                GB_LSHIFT (Z_iso_code , 16) |  // 0 or 1       1

                // mask (1 hex digit)
                GB_LSHIFT (mask_ecode , 12) |  // 0 to 13      4

                // type of R (1 hex digit)
                GB_LSHIFT (rcode      ,  8) |  // 0 to 14      4

                // sparsity structures of R, M, C, and Z (2 hex digits)
                GB_LSHIFT (rsparsity  ,  6) |  // 0 to 3       2
                GB_LSHIFT (csparsity  ,  4) |  // 0 to 3       2
                GB_LSHIFT (msparsity  ,  2) |  // 0 to 3       2
                GB_LSHIFT (zsparsity  ,  0) ;  // 0 to 3       2
}

