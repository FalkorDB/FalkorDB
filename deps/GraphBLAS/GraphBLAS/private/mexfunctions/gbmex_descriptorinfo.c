//------------------------------------------------------------------------------
// gbmex_descriptorinfo: print a GraphBLAS descriptor (for illustration only)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// gbmex_descriptorinfo
// gbmex_descriptorinfo (desc)

#include "gb_interface.h"
#include "gbmx_interface.h"

#undef  FREE_ALL
#define FREE_ALL GrB_Descriptor_free (&desc) ;

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
    // check inputs (no outputs to construct)
    //--------------------------------------------------------------------------

    GrB_Descriptor desc = NULL ;
    int arena = GrB_DEFAULT ;   // use default arena for temporary workspace

    GBMX_USAGE (nargin <= 1 && nargout == 0, USAGE) ;

    //--------------------------------------------------------------------------
    // find the arguments
    //--------------------------------------------------------------------------

    struct gb_descriptor_struct gbdesc ;
    gbmx_mxarray_to_descriptor (&gbdesc, (nargin == 0) ? NULL : pargin [0]) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the GrB_Descriptor
    //--------------------------------------------------------------------------

    gbdesc.nondefault = true ;      // ensure the GrB_Descriptor is allocated
    OK (gb_get_descriptor (&desc, &gbdesc, arena, err)) ;
    ASSERT (desc != NULL) ;

    //--------------------------------------------------------------------------
    // print the GraphBLAS descriptor
    //--------------------------------------------------------------------------

    OK (GxB_Descriptor_fprint (desc, "", GxB_COMPLETE, NULL)) ;

    //--------------------------------------------------------------------------
    // print the extra terms in the interface descriptor
    //--------------------------------------------------------------------------

    printf ("    d.kind     = ") ;
    switch (gbdesc.kind)
    {
        case KIND_SPARSE  : printf ("sparse\n")  ; break ;
        case KIND_FULL    : printf ("full\n")    ; break ;
        case KIND_BUILTIN : printf ("builtin\n") ; break ;
        case KIND_GRB     :
        default           : printf ("GrB\n")     ; break ;
    }

    printf ("    d.base     = ") ;
    switch (gbdesc.base)
    {
        case BASE_0_INT    : printf ("zero-based\n")    ; break ;
        case BASE_1_INT    : printf ("one-based int\n") ; break ;
        case BASE_1_DOUBLE : printf ("one-based\n")     ; break ;
        case BASE_DEFAULT  :
        default            : printf ("default (one-based int)\n") ; break ;
    }

    printf ("    d.format   = ") ;

    switch (gbdesc.sparsity)
    {
        case GxB_HYPERSPARSE :                              // 1
            printf ("hypersparse ") ;
            break ;
        case GxB_SPARSE :                                   // 2
            printf ("sparse ") ;
            break ;
        case GxB_HYPERSPARSE + GxB_SPARSE :                 // 3
            printf ("sparse/hypersparse") ;
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
            printf ("sparse/hypersparse/bitmap ") ;
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
        case GxB_HYPERSPARSE + GxB_SPARSE + GxB_FULL :      // 11
            printf ("sparse/hypersparse/full ") ;
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
        default :
        case GxB_HYPERSPARSE + GxB_SPARSE + GxB_BITMAP + GxB_FULL : // 15
            printf ("sparse/hypersparse/bitmap/full ") ;
            break ;
    }

    switch (gbdesc.fmt)
    {
        case GxB_BY_ROW    : printf ("by row\n")     ; break ;
        case GxB_BY_COL    : printf ("by col\n")     ; break ;
        case GxB_NO_FORMAT :
        default            : printf ("by default\n") ; break ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_ALL ;
    gb_wrapup ( ) ;
}

