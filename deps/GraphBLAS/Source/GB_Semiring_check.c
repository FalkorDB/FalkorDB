//------------------------------------------------------------------------------
// GB_Semiring_check: check and print a semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GB_Semiring_check          // check a GraphBLAS semiring
(
    const GrB_Semiring semiring,    // GraphBLAS semiring to print and check
    const char *name,               // name of the semiring, optional
    int pr,                         // print level
    FILE *f                         // file for output
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GBPR0 ("\n    GraphBLAS Semiring: %s ", ((name != NULL) ? name : "")) ;

    if (semiring == NULL)
    { 
        GBPR0 ("NULL\n") ;
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // check object
    //--------------------------------------------------------------------------

    GB_CHECK_MAGIC (semiring) ;
    GBPR0 (semiring->header_size > 0 ? "(user-defined):" : "(built-in):") ;

    GrB_Monoid add = semiring->add ;
    GrB_BinaryOp mult = semiring->multiply ;

    if (semiring->name == NULL)
    { 
        // semiring contains a built-in monoid and multiply operator
        char *add_name  = (add  == NULL) ? "nil" : add->op->name ;
        char *mult_name = (mult == NULL) ? "nil" : mult->name ;
        GBPR0 (" (%s,%s)", add_name, mult_name) ;
    }
    else
    { 
        // semiring contains user-defined monoid and/or multiply operator
        GBPR0 (" (%s)", semiring->name) ;
        if (semiring->name_len != strlen (semiring->name))
        { 
            GBPR0 ("    Semiring->name invalid\n") ;
            return (GrB_INVALID_OBJECT) ;
        }
    }

    GrB_Info info ;
    info = GB_Monoid_check (add, "semiring->add", pr, f) ;
    if (info != GrB_SUCCESS)
    { 
        GBPR0 ("    Semiring->add invalid\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    info = GB_BinaryOp_check (mult, "semiring->multiply", pr, f) ;
    if (info != GrB_SUCCESS)
    { 
        GBPR0 ("    Semiring->multiply invalid\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    // z = multiply(x,y); type of z must match monoid type
    if (mult->ztype != add->op->ztype)
    { 
        GBPR0 ("    Semiring multiply output domain must match monoid"
            " domain\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    return (GrB_SUCCESS) ;
}

