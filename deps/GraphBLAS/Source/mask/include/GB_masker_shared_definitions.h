//------------------------------------------------------------------------------
// GB_masker_shared_definitions.h: common macros for masker kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_masker_shared_definitions.h provides default definitions for all masker
// kernels, if the special cases have not been #define'd prior to #include'ing
// this file.  This file is shared by generic and both CPU and CUDA JIT
// kernels.  There are no factory masker kernels.

#include "include/GB_kernel_shared_definitions.h"

#ifndef GB_MASKER_SHARED_DEFINITIONS_H
#define GB_MASKER_SHARED_DEFINITIONS_H

// the type of R, M, C, and Z
#ifndef GB_R_TYPE
#define GB_R_TYPE GB_void
#endif

// copy C(i,j) to R(i,j)
#ifndef GB_COPY_C_TO_R
#define GB_COPY_C_TO_R(Rx,pR,Cx,pC,C_iso,rsize) \
        memcpy (Rx +(pR)*rsize, Cx +(C_iso ? 0:(pC)*rsize), rsize) ;
#endif

// copy Z(i,j) to R(i,j)
#ifndef GB_COPY_Z_TO_R
#define GB_COPY_Z_TO_R(Rx,pR,Zx,pZ,Z_iso,rsize) \
        memcpy (Rx +(pR)*rsize, Zx +(Z_iso ? 0:(pZ)*rsize), rsize) ;
#endif

// copy a range of values from C to R
#ifndef GB_COPY_C_TO_R_RANGE
#define GB_COPY_C_TO_R_RANGE(Rx,pR,Cx,pC,C_iso,rsize,cjnz)          \
{                                                                   \
    if (C_iso)                                                      \
    {                                                               \
        for (int64_t k = 0 ; k < cjnz ; k++)                        \
        {                                                           \
            /* Rx [pR+k] = Cx [0] */                                \
            GB_COPY_C_TO_R (Rx, pR+k, Cx, 0, true, rsize) ;         \
        }                                                           \
    }                                                               \
    else                                                            \
    {                                                               \
        /* Rx [pR:pR+cjnz-1] = Cx [pC:pC+cjnz-1] */                 \
        memcpy (Rx +(pR)*rsize, Cx +(pC)*rsize, (cjnz)*rsize) ;     \
    }                                                               \
}
#endif

// copy a range of values from Z to R
#ifndef GB_COPY_Z_TO_R_RANGE
#define GB_COPY_Z_TO_R_RANGE(Rx,pR,Zx,pZ,Z_iso,rsize,zjnz)          \
{                                                                   \
    if (Z_iso)                                                      \
    {                                                               \
        for (int64_t k = 0 ; k < zjnz ; k++)                        \
        {                                                           \
            /* Rx [pR+k] = Zx [0] */                                \
            GB_COPY_Z_TO_R (Rx, pR+k, Zx, 0, true, rsize) ;         \
        }                                                           \
    }                                                               \
    else                                                            \
    {                                                               \
        /* Rx [pR:pR+zjnz-1] = Zx [pZ:pZ+zjnz-1] */                 \
        memcpy (Rx +(pR)*rsize, Zx +(pZ)*rsize, (zjnz)*rsize) ;     \
    }                                                               \
}
#endif

#ifndef GB_MASK_COMP
#define GB_MASK_COMP Mask_comp
#endif

#ifndef GB_MASK_STRUCT
#define GB_MASK_STRUCT Mask_struct
#endif

#ifndef GB_NO_MASK
#define GB_NO_MASK 0
#endif

#ifndef GB_R_IS_BITMAP
#define GB_R_IS_BITMAP (R_sparsity == GxB_BITMAP)
#endif
#ifndef GB_R_IS_FULL
#define GB_R_IS_FULL false
#endif
#ifndef GB_R_IS_SPARSE
#define GB_R_IS_SPARSE (R_sparsity== GxB_SPARSE)
#endif
#ifndef GB_R_IS_HYPER
#define GB_R_IS_HYPER (R_sparsity == GxB_HYPERSPARSE)
#endif

#ifndef GB_C_ISO
#define GB_C_ISO C_iso
#endif

#ifndef GB_Z_IS_BITMAP
#define GB_Z_IS_BITMAP Z_is_bitmap
#endif
#ifndef GB_Z_IS_FULL
#define GB_Z_IS_FULL Z_is_full
#endif
#ifndef GB_Z_ISO
#define GB_Z_ISO Z_iso
#endif

#endif

