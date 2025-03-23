//------------------------------------------------------------------------------
// GB_masker_phase2_jit: construct R = masker (C,M,Z)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

typedef GB_JIT_KERNEL_MASKER_PHASE2_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_masker_phase2_jit       // phase2 for R = masker (C,M,Z)
(
    GrB_Matrix R,                   // output matrix, static header
    // tasks from phase1a:
    const GB_task_struct *restrict TaskList,     // array of structs
    const int R_ntasks,               // # of tasks
    const int R_nthreads,             // # of threads to use
    // analysis from phase0:
    const int64_t *restrict R_to_M,
    const int64_t *restrict R_to_C,
    const int64_t *restrict R_to_Z,
    // original input:
    const GrB_Matrix M,             // required mask
    const bool Mask_comp,           // if true, then M is complemented
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix C,
    const GrB_Matrix Z,
    const int64_t *restrict C_ek_slicing,
    const int C_nthreads,
    const int C_ntasks,
    const int64_t *restrict M_ek_slicing,
    const int M_nthreads,
    const int M_ntasks
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_masker (&encoding, &suffix,
        GB_JIT_KERNEL_MASKER_PHASE2, R, R->p_is_32, R->j_is_32, R->i_is_32,
        M, Mask_struct, Mask_comp, C, Z) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_masker_family, "masker_phase2",
        hash, &encoding, suffix, NULL, NULL, NULL, R->type, NULL, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    #include "include/GB_pedantic_disable.h"
    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (R, TaskList, R_ntasks, R_nthreads,
        R_to_M, R_to_C, R_to_Z, M, Mask_comp, Mask_struct, C, Z,
        C_ek_slicing, C_ntasks, C_nthreads,
        M_ek_slicing, M_ntasks, M_nthreads, &GB_callback)) ;
}

