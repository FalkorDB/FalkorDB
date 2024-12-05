//------------------------------------------------------------------------------
// GB_masker_phase1_jit: find # of entries in R = masker (C,M,Z)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

typedef GB_JIT_KERNEL_MASKER_PHASE1_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_masker_phase1_jit       // count nnz in each R(:,j)
(
    // computed by phase1:
    int64_t *Rp,                    // output of size Rnvec+1
    int64_t *Rnvec_nonempty,        // # of non-empty vectors in R
    // tasks from phase1a:
    GB_task_struct *restrict TaskList,       // array of structs
    const int R_ntasks,               // # of tasks
    const int R_nthreads,             // # of threads to use
    // analysis from phase0:
    const int64_t Rnvec,
    const int64_t *restrict Rh,
    const int64_t *restrict R_to_M,
    const int64_t *restrict R_to_C,
    const int64_t *restrict R_to_Z,
    // original input:
    const GrB_Matrix M,             // required mask
    const bool Mask_comp,           // if true, then M is complemented
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix C,
    const GrB_Matrix Z
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_masker (&encoding, &suffix,
        GB_JIT_KERNEL_MASKER_PHASE1, NULL, M, Mask_struct, Mask_comp, C, Z) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_masker_family, "masker_phase1",
        hash, &encoding, suffix, NULL, NULL, NULL, NULL, NULL, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (Rp, Rnvec_nonempty, TaskList, R_ntasks, R_nthreads,
        Rnvec, Rh, R_to_M, R_to_C, R_to_Z, M, Mask_comp, Mask_struct, C, Z)) ;
}

