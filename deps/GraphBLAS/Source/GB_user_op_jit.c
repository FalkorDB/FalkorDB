//------------------------------------------------------------------------------
// GB_user_op_jit: construct a user operator in its own JIT kernel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

typedef GB_JIT_KERNEL_USER_OP_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_user_op_jit         // construct a user operator in a JIT kernel
(
    // output:
    void **user_function,       // function pointer
    // input:
    const GB_Operator op        // unary, index unary, or binary op
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_user_op (&encoding, &suffix, op) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_user_op_family, "user_op",
        hash, &encoding, suffix, NULL, NULL,
        op, NULL, NULL, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    char *ignore ;
    return (GB_jit_kernel (user_function, &ignore)) ;
}

