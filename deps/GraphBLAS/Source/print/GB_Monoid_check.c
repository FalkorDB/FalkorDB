//------------------------------------------------------------------------------
// GB_Monoid_check: check and print a monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "get_set/GB_get_set.h"

GrB_Info GB_Monoid_check        // check a GraphBLAS monoid
(
    const GrB_Monoid monoid,    // GraphBLAS monoid to print and check
    const char *name,           // name of the monoid, optional
    int pr,                     // print level
    FILE *f,                    // file for output
    bool in_semiring            // if true, then called by GB_Semiring_check
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GBPR0 ("\n    GraphBLAS Monoid: %s ", ((name != NULL) ? name : "")) ;

    if (monoid == NULL)
    { 
        GBPR0 ("NULL\n") ;
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // check object
    //--------------------------------------------------------------------------

    GB_CHECK_MAGIC (monoid) ;
    if (monoid->header_size == 0)
    { 
        GBPR0 ("(built-in):") ;
    }
    else if (monoid->hash == 0)
    { 
        GBPR0 ("(user-defined; same as built-in):") ;
    }
    else
    { 
        GBPR0 ("(user-defined):") ;
    }

    GrB_Info info = GB_BinaryOp_check (monoid->op, "monoid->op", pr, f) ;
    if (info != GrB_SUCCESS || GB_OP_IS_POSITIONAL (monoid->op))
    { 
        GBPR0 ("    Monoid contains an invalid operator\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    // name given by GrB_set, or 'GrB_*' name for built-in objects
    const char *given_name = GB_monoid_name_get (monoid) ;
    if (given_name != NULL)
    { 
        GBPR0 ("    Monoid given name: [%s]\n", given_name) ;
    }

    if (monoid->op->xtype != monoid->op->ztype ||
        monoid->op->ytype != monoid->op->ztype)
    { 
        GBPR0 ("    All domains of operator must be the same\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    if (monoid->identity == NULL)
    { 
        GBPR0 ("    Identity value is missing\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    // print the identity and terminal values
    if (pr != GxB_SILENT)
    { 
        char *string = NULL ;
        size_t string_size = 0 ;

        // print the identity value, if present
        GBPR ("    identity: [ ") ;
        info = GB_entry_check (monoid->op->ztype, monoid->identity, pr, f,
            &string, &string_size) ;
        GB_FREE_MEMORY (&string, string_size) ;
        if (info != GrB_SUCCESS) return (info) ;
        GBPR (" ] ") ;

        // print the terminal value, if present
        if (monoid->terminal != NULL)
        { 
            GBPR ("terminal: [ ") ;
            info = GB_entry_check (monoid->op->ztype, monoid->terminal, pr, f,
                &string, &string_size) ;
            GB_FREE_MEMORY (&string, string_size) ;
            if (info != GrB_SUCCESS) return (info) ;
            GBPR (" ]") ;
        }
        if (!in_semiring) GBPR ("\n") ;
    }

    return (GrB_SUCCESS) ;
}

