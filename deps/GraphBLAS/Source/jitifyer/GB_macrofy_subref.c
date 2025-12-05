//------------------------------------------------------------------------------
// GB_macrofy_subref: construct all macros for subref methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

void GB_macrofy_subref          // construct all macros for GrB_extract
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t method_code,
    GrB_Type ctype
)
{

    //--------------------------------------------------------------------------
    // extract the subref method_code
    //--------------------------------------------------------------------------

    // R integer sizes and sparsity
    bool Rp_is_32    = GB_RSHIFT (method_code, 27, 1) ;
    bool Rj_is_32    = GB_RSHIFT (method_code, 26, 1) ;
    bool Ri_is_32    = GB_RSHIFT (method_code, 25, 1) ;
    int rsparsity    = GB_RSHIFT (method_code, 23, 2) ;

    // C, A integer sizes (2 hex digits)
    bool Cp_is_32    = GB_RSHIFT (method_code, 21, 1) ;
    bool Cj_is_32    = GB_RSHIFT (method_code, 20, 1) ;
    bool Ci_is_32    = GB_RSHIFT (method_code, 19, 1) ;

    bool Ap_is_32    = GB_RSHIFT (method_code, 18, 1) ;
    bool Aj_is_32    = GB_RSHIFT (method_code, 17, 1) ;
    bool Ai_is_32    = GB_RSHIFT (method_code, 16, 1) ;

    // need_qsort, I and J bits (1 hex digit)
    bool I_is_32     = GB_RSHIFT (method_code, 15, 1) ;
    bool J_is_32     = GB_RSHIFT (method_code, 14, 1) ;
    // 13: unused
    int needqsort    = GB_RSHIFT (method_code, 12, 1) ;

    // Ikind, Jkind (1 hex digit)
    int Ikind        = GB_RSHIFT (method_code, 10, 2) ;
    int Jkind        = GB_RSHIFT (method_code,  8, 2) ;

    // type of C and A (1 hex digit)
//  int ccode        = GB_RSHIFT (method_code,  4, 4) ;

    // sparsity structures of C and A (1 hex digit)
    int csparsity    = GB_RSHIFT (method_code,  2, 2) ;
    int asparsity    = GB_RSHIFT (method_code,  0, 2) ;

    //--------------------------------------------------------------------------
    // describe the subref
    //--------------------------------------------------------------------------

    fprintf (fp, "// subref: C=A(I,J) where C and A are %s\n",
        (asparsity <= 1) ? "sparse/hypersparse" : "bitmap/full") ;

    fprintf (fp, "#define GB_I_KIND ") ;
    switch (Ikind)
    {
        case GB_ALL    : fprintf (fp, "GB_ALL\n"    ) ; break ;
        case GB_RANGE  : fprintf (fp, "GB_RANGE\n"  ) ; break ;
        case GB_STRIDE : fprintf (fp, "GB_STRIDE\n" ) ; break ;
        case GB_LIST   : fprintf (fp, "GB_LIST\n"   ) ; break ;
        default:;
    }
    fprintf (fp, "#define GB_I_TYPE uint%d_t\n", I_is_32 ? 32 : 64) ;
    if (asparsity <= 1)
    { 
        // C and A are sparse/hypersparse
        // Jkind not needed for sparse subsref
        fprintf (fp, "#define GB_NEED_QSORT %d\n", needqsort) ;
    }
    else
    { 
        // C and A are bitmap/full
        // need_qsort not needed for bitmap subsref
        fprintf (fp, "#define GB_J_KIND ") ;
        switch (Jkind)
        {
            case GB_ALL    : fprintf (fp, "GB_ALL\n"    ) ; break ;
            case GB_RANGE  : fprintf (fp, "GB_RANGE\n"  ) ; break ;
            case GB_STRIDE : fprintf (fp, "GB_STRIDE\n" ) ; break ;
            case GB_LIST   : fprintf (fp, "GB_LIST\n"   ) ; break ;
            default:;
        }
        fprintf (fp, "#define GB_J_TYPE uint%d_t\n", J_is_32 ? 32 : 64) ;
    }

    //--------------------------------------------------------------------------
    // construct the typedefs
    //--------------------------------------------------------------------------

    GB_macrofy_typedefs (fp, ctype, NULL, NULL, NULL, NULL, NULL, NULL) ;

    //--------------------------------------------------------------------------
    // construct the macros for C, A, and R
    //--------------------------------------------------------------------------

    GB_macrofy_sparsity (fp, "C", csparsity) ;
    GB_macrofy_nvals (fp, "C", csparsity, false) ;
    GB_macrofy_type (fp, "C", "_", ctype->name) ;
    GB_macrofy_bits (fp, "C", Cp_is_32, Cj_is_32, Ci_is_32) ;

    GrB_Type atype = ctype ;        // C and A have the same type
    GB_macrofy_sparsity (fp, "A", asparsity) ;
    GB_macrofy_nvals (fp, "A", asparsity, false) ;
    GB_macrofy_type (fp, "A", "_", atype->name) ;
    GB_macrofy_bits (fp, "A", Ap_is_32, Aj_is_32, Ai_is_32) ;

    // R is always GrB_UINT64, and iso-valued (its values are not used)
    GB_macrofy_sparsity (fp, "R", rsparsity) ;
    GB_macrofy_nvals (fp, "R", rsparsity, false) ;
    GB_macrofy_bits (fp, "R", Rp_is_32, Rj_is_32, Ri_is_32) ;

    //--------------------------------------------------------------------------
    // include the final default definitions
    //--------------------------------------------------------------------------

    fprintf (fp, "\n#include \"include/GB_kernel_shared_definitions.h\"\n") ;
}

