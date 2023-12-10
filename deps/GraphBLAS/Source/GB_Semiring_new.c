//------------------------------------------------------------------------------
// GB_Semiring_new: create a new semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The semiring struct is already allocated on input.

#include "GB.h"
#include "GB_Semiring_new.h"
#include "GB_jitifyer.h"

GrB_Info GB_Semiring_new            // create a semiring
(
    GrB_Semiring semiring,          // semiring to create
    GrB_Monoid add,                 // additive monoid of the semiring
    GrB_BinaryOp multiply           // multiply operator of the semiring
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (semiring != NULL) ;
    ASSERT (add != NULL) ;
    ASSERT (multiply != NULL) ;
    ASSERT_MONOID_OK (add, "semiring->add", GB0) ;
    ASSERT_BINARYOP_OK (multiply, "semiring->multiply", GB0) ;

    //--------------------------------------------------------------------------
    // create the semiring
    //--------------------------------------------------------------------------

    // z = multiply(x,y); type of z must match monoid z = add(z,z)
    if (multiply->ztype != add->op->ztype)
    { 
        return (GrB_DOMAIN_MISMATCH) ;
    }

    // initialize the semiring
    semiring->magic = GB_MAGIC ;
    semiring->user_name = NULL ;            // user_name for GrB_get/GrB_set
    semiring->user_name_size = 0 ;
    semiring->add = add ;
    semiring->multiply = multiply ;
    semiring->name = NULL ;
    if (add->hash == 0 && multiply->hash == 0)
    { 
        // semiring consists of builtin types and operators only;
        // no need for the semiring name or hash
        semiring->hash = 0 ;
    }
    else
    {
        // construct the semiring name
        size_t add_len  = strlen (semiring->add->op->name) ;
        size_t mult_len = strlen (semiring->multiply->name) ;
        semiring->name_len = (int32_t) (add_len + mult_len + 1) ;
        semiring->name = GB_MALLOC (semiring->name_len + 1, char,
            &(semiring->name_size)) ;
        if (semiring->name == NULL)
        { 
            // out of memory
            return (GrB_OUT_OF_MEMORY) ;
        }
        char *p = semiring->name ;
        memcpy (p, semiring->add->op->name, add_len) ;
        p += add_len ;
        (*p++) = '_' ;
        memcpy (p, semiring->multiply->name, mult_len) ;
        p += mult_len ;
        (*p) = '\0' ;
        // construct the semiring hash from the newly created name.
        // the semiring is JIT'able only if its 2 operators are JIT'able.
        semiring->hash = GB_jitifyer_hash (semiring->name, semiring->name_len,
            add->hash != UINT64_MAX && multiply->hash != UINT64_MAX) ;
    }
    ASSERT_SEMIRING_OK (semiring, "new semiring", GB0) ;
    return (GrB_SUCCESS) ;
}

