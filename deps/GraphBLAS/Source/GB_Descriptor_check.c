//------------------------------------------------------------------------------
// GB_Descriptor_check: check and print a Descriptor
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

//------------------------------------------------------------------------------
// GB_dc: check a single descriptor field
//------------------------------------------------------------------------------

static GrB_Info GB_dc
(
    int kind,                           // 0, 1, or 2
    const char *field,
    const GrB_Desc_Value v,
    const GrB_Desc_Value nondefault,    // for kind == 0
    int pr,                             // print level
    FILE *f 
)
{

    bool ok = true ;
    GrB_Info info = GrB_SUCCESS ;

    GBPR0 ("    d.%s = ", field) ;
    switch ((int) v)
    {
        case GxB_DEFAULT            : GBPR0 ("default   ") ; break ;
        case GrB_COMP               : GBPR0 ("complement") ; break ;
        case GrB_STRUCTURE          : GBPR0 ("structure ") ; break ;
        case GrB_COMP+GrB_STRUCTURE : GBPR0 ("structural complement") ; break ;
        case GrB_TRAN               : GBPR0 ("transpose ") ; break ;
        case GrB_REPLACE            : GBPR0 ("replace   ") ; break ;
        case GxB_AxB_SAXPY          : GBPR0 ("saxpy     ") ; break ;
        case GxB_AxB_GUSTAVSON      : GBPR0 ("Gustavson ") ; break ;
        case GxB_AxB_HASH           : GBPR0 ("hash      ") ; break ;
        case GxB_AxB_DOT            : GBPR0 ("dot       ") ; break ;
        default                     : GBPR0 ("unknown   ") ;
            info = GrB_INVALID_OBJECT ;
            ok = false ;
            break ;
    }

    if (ok)
    {
        if (kind == 0)
        {
            // most descriptor fields can be set to the default,
            // or just one non-default value
            if (! (v == GxB_DEFAULT || v == nondefault))
            { 
                ok = false ;
            }
        }
        else if (kind == 1)
        {
            // mask: can only be one of 4 different values
            if (! (v == GxB_DEFAULT || v == GrB_COMP || v == GrB_STRUCTURE ||
                   v == (GrB_COMP + GrB_STRUCTURE)))
            { 
                ok = false ;
            }
        }
        else // kind == 2
        {
            // GxB_AxB_METHOD:
            if (! (v == GxB_DEFAULT || v == GxB_AxB_GUSTAVSON
                || v == GxB_AxB_DOT || v == GxB_AxB_HASH || v == GxB_AxB_SAXPY))
            { 
                ok = false ;
            }
        }
    }

    if (!ok)
    { 
        GBPR0 (" (invalid value for this field)") ;
        info = GrB_INVALID_OBJECT ;
    }

    GBPR0 ("\n") ;

    return (info) ;
}

//------------------------------------------------------------------------------
// GB_Descriptor_check
//------------------------------------------------------------------------------

GrB_Info GB_Descriptor_check    // check a GraphBLAS descriptor
(
    const GrB_Descriptor D,     // GraphBLAS descriptor to print and check
    const char *name,           // name of the descriptor, optional
    int pr,                     // print level
    FILE *f                     // file for output
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GBPR0 ("\n    GraphBLAS Descriptor: %s ", ((name != NULL) ? name : "")) ;

    if (D == NULL)
    { 
        GBPR0 ("NULL\n") ;
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // check object
    //--------------------------------------------------------------------------

    GB_CHECK_MAGIC (D) ;

    GBPR0 ("\n") ;

    GrB_Info info [5] ;
    info [0] = GB_dc (0, "out     ", D->out,  GrB_REPLACE, pr, f) ;
    info [1] = GB_dc (1, "mask    ", D->mask, GxB_DEFAULT, pr, f) ;
    info [2] = GB_dc (0, "in0     ", D->in0,  GrB_TRAN,    pr, f) ;
    info [3] = GB_dc (0, "in1     ", D->in1,  GrB_TRAN,    pr, f) ;
    info [4] = GB_dc (2, "axb     ", D->axb,  GxB_DEFAULT, pr, f) ;

    for (int i = 0 ; i < 5 ; i++)
    {
        if (info [i] != GrB_SUCCESS)
        { 
            GBPR0 ("    Descriptor field set to an invalid value\n") ;
            return (GrB_INVALID_OBJECT) ;
        }
    }

    if (D->do_sort)
    { 
        GBPR0 ("    d.sort     = true\n") ;
    }

    if (D->import != GxB_DEFAULT)
    { 
        GBPR0 ("    d.import   = secure\n") ;
    }

    if (D->compression != GxB_DEFAULT)
    { 
        GBPR0 ("    d.compression = %d\n", D->compression) ;
    }

    return (GrB_SUCCESS) ;
}

