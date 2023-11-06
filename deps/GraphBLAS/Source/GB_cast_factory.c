//------------------------------------------------------------------------------
// GB_cast_factory: return a pointer to a typecasting function
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Returns a pointer to a function f(z,x,s) that copies its input x into its
// output z, casting as needed.  That is, it computes z = (type of z) x.
// s is the size for user-defined types, which can only be copied.

// This function returns one of ((13*13) + 1) pointers to a typecasting/copy
// function, defined by GB_casting.c.  The last function pointer is the
// function GB_copy_user_user.

#include "GB.h"

GB_cast_function GB_cast_factory   // returns pointer to function to cast x to z
(
    const GB_Type_code code1,      // the type of z, the output value
    const GB_Type_code code2       // the type of x, the input value
)
{ 

    //--------------------------------------------------------------------------
    // define the worker for the switch factory
    //--------------------------------------------------------------------------

    // the worker selects a typecast function and returns it to the caller
    #define GB_WORKER(ignore1,ignore2,ztype,ignore3,xtype) \
        return (&GB (_cast_ ## ztype ## _ ## xtype)) ;

    //--------------------------------------------------------------------------
    // launch the switch factory
    //--------------------------------------------------------------------------

    // switch factory for two built-in types; user types are skipped.
    // no generic worker so the switch factory cannot be disabled.
    // no JIT kernel.
    #include "GB_twotype_factory.c"

    //--------------------------------------------------------------------------
    // user-defined types fall through the switch factory to here
    //--------------------------------------------------------------------------

    return (&GB_copy_user_user) ;
}

