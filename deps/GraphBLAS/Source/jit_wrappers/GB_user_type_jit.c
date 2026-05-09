//------------------------------------------------------------------------------
// GB_user_type_jit: construct a user type in its own JIT kernel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

typedef GB_JIT_KERNEL_USER_TYPE_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_user_type_jit       // construct a user type in a JIT kernel
(
    // output:
    uint64_t *user_type_memsize,     // sizeof the user type
    // input:
    const GrB_Type type         // user-defined type
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_user_type (&encoding, &suffix, type) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_user_type_family, "user_type",
        hash, &encoding, suffix, NULL, NULL,
        NULL, type, NULL, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    #include "include/GB_pedantic_disable.h"
    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    char *ignore ;
    return (GB_jit_kernel (user_type_memsize, &ignore)) ;
}

