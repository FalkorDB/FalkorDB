//------------------------------------------------------------------------------
// GB_macrofy_subref: construct all macros for subref methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
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

    // need_qsort, I_has_duplicates (1 hex digit)
    int ihasdupl    = GB_RSHIFT (method_code, 13, 1) ;
    int needqsort   = GB_RSHIFT (method_code, 12, 1) ;

    // Ikind, Jkind (1 hex digit)
    int Ikind       = GB_RSHIFT (method_code, 10, 2) ;
    int Jkind       = GB_RSHIFT (method_code,  8, 2) ;

    // type of C and A (1 hex digit)
    int ccode       = GB_RSHIFT (method_code,  4, 4) ;

    // sparsity structures of C and A (1 hex digit)
    int csparsity   = GB_RSHIFT (method_code,  2, 2) ;
    int asparsity   = GB_RSHIFT (method_code,  0, 2) ;

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
    if (asparsity <= 1)
    { 
        // C and A are sparse/hypersparse
        // Jkind not needed for sparse subsref
        fprintf (fp, "#define GB_NEED_QSORT %d\n", needqsort) ;
        fprintf (fp, "#define GB_I_HAS_DUPLICATES %d\n", ihasdupl) ;
    }
    else
    { 
        // C and A are bitmap/full
        // need_qsort, I_has_duplicates not needed for bitmap subsref
        fprintf (fp, "#define GB_J_KIND ") ;
        switch (Jkind)
        {
            case GB_ALL    : fprintf (fp, "GB_ALL\n"    ) ; break ;
            case GB_RANGE  : fprintf (fp, "GB_RANGE\n"  ) ; break ;
            case GB_STRIDE : fprintf (fp, "GB_STRIDE\n" ) ; break ;
            case GB_LIST   : fprintf (fp, "GB_LIST\n"   ) ; break ;
            default:;
        }
    }

    //--------------------------------------------------------------------------
    // construct the typedefs
    //--------------------------------------------------------------------------

    GB_macrofy_typedefs (fp, ctype, NULL, NULL, NULL, NULL, NULL) ;

    //--------------------------------------------------------------------------
    // construct the macros for C and A
    //--------------------------------------------------------------------------

    GB_macrofy_sparsity (fp, "C", csparsity) ;
    GB_macrofy_nvals (fp, "C", csparsity, false) ;
    GB_macrofy_type (fp, "C", "_", ctype->name) ;

    GrB_Type atype = ctype ;        // C and A have the same type
    GB_macrofy_sparsity (fp, "A", asparsity) ;
    GB_macrofy_nvals (fp, "A", asparsity, false) ;
    GB_macrofy_type (fp, "A", "_", atype->name) ;

    //--------------------------------------------------------------------------
    // include the final default definitions
    //--------------------------------------------------------------------------

    fprintf (fp, "\n#include \"include/GB_kernel_shared_definitions.h\"\n") ;
}

