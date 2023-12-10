//------------------------------------------------------------------------------
// gbdescriptorinfo: print a GraphBLAS descriptor (for illustration only)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbdescriptorinfo
// gbdescriptorinfo (desc)

#include "gb_interface.h"

#define USAGE "usage: GrB.descriptorinfo or GrB.descriptorinfo (desc)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin <= 1 && nargout == 0, USAGE) ;

    //--------------------------------------------------------------------------
    // construct the GraphBLAS descriptor
    //--------------------------------------------------------------------------

    base_enum_t base = BASE_DEFAULT ;
    kind_enum_t kind = KIND_GRB ;
    GxB_Format_Value fmt = GxB_NO_FORMAT ;
    int sparsity = 0 ;
    GrB_Descriptor desc = NULL ;
    if (nargin > 0)
    {
        desc = gb_mxarray_to_descriptor (pargin [nargin-1], &kind, &fmt,
            &sparsity, &base) ;
    }

    if (desc == NULL)
    { 
        printf ("\nDefault GraphBLAS descriptor:\n") ;
        OK (GrB_Descriptor_new (&desc)) ;
    }

    //--------------------------------------------------------------------------
    // print the GraphBLAS descriptor
    //--------------------------------------------------------------------------

    OK (GxB_Descriptor_fprint (desc, "", GxB_COMPLETE, NULL)) ;

    //--------------------------------------------------------------------------
    // print the extra terms in the interface descriptor
    //--------------------------------------------------------------------------

    printf ("    d.kind     = ") ;
    switch (kind)
    {
        case KIND_SPARSE  : printf ("sparse\n")  ; break ;
        case KIND_FULL    : printf ("full\n")    ; break ;
        case KIND_BUILTIN : printf ("builtin\n") ; break ;
        case KIND_GRB     :
        default           : printf ("GrB\n")     ; break ;
    }

    printf ("    d.base     = ") ;
    switch (base)
    {
        case BASE_0_INT64  : printf ("zero-based\n")    ; break ;
        case BASE_1_INT64  : printf ("one-based int\n") ; break ;
        case BASE_1_DOUBLE : printf ("one-based\n")     ; break ;
        case BASE_DEFAULT  :
        default            : printf ("default\n")       ; break ;
    }

    printf ("    d.format   = ") ;

    switch (sparsity)
    {
        case GxB_HYPERSPARSE :                              // 1
            printf ("hypersparse ") ;
            break ;
        case GxB_SPARSE :                                   // 2
            printf ("sparse ") ;
            break ;
        case GxB_HYPERSPARSE + GxB_SPARSE :                 // 3
            printf ("hypersparse/sparse ") ;
            break ;
        case GxB_BITMAP :                                   // 4
            printf ("bitmap ") ;
            break ;
        case GxB_HYPERSPARSE + GxB_BITMAP :                 // 5
            printf ("hypersparse/bitmap ") ;
            break ;
        case GxB_SPARSE + GxB_BITMAP :                      // 6
            printf ("sparse/bitmap ") ;
            break ;
        case GxB_HYPERSPARSE + GxB_SPARSE + GxB_BITMAP :    // 7
            printf ("hypersparse/sparse/bitmap ") ;
            break ;
        case GxB_FULL :                                     // 8
            printf ("full ") ;
            break ;
        case GxB_HYPERSPARSE + GxB_FULL :                   // 9
            printf ("hypersparse/full ") ;
            break ;
        case GxB_SPARSE + GxB_FULL :                        // 10
            printf ("sparse/full ") ;
            break ;
        default :
        case GxB_HYPERSPARSE + GxB_SPARSE + GxB_FULL :      // 11
            // printf ("hypersparse/sparse/full ") ;
            break ;
        case GxB_BITMAP + GxB_FULL :                        // 12
            printf ("bitmap/full ") ;
            break ;
        case GxB_HYPERSPARSE + GxB_BITMAP + GxB_FULL :      // 13
            printf ("hypersparse/bitmap/full ") ;
            break ;
        case GxB_SPARSE + GxB_BITMAP + GxB_FULL :           // 14
            printf ("sparse/bitmap/full ") ;
            break ;
        case GxB_HYPERSPARSE + GxB_SPARSE + GxB_BITMAP + GxB_FULL : // 15
            printf ("hypersparse/sparse/bitmap/full ") ;
            break ;
    }

    switch (fmt)
    {
        case GxB_BY_ROW    : printf ("by row\n")     ; break ;
        case GxB_BY_COL    : printf ("by col\n")     ; break ;
        case GxB_NO_FORMAT :
        default            : printf ("by default\n") ; break ;
    }

    //--------------------------------------------------------------------------
    // free the descriptor
    //--------------------------------------------------------------------------

    OK (GrB_Descriptor_free (&desc)) ;
    GB_WRAPUP ;
}

