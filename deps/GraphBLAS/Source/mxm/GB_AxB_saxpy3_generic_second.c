//------------------------------------------------------------------------------
// GB_AxB_saxpy3_generic_second.c: C=A*B, C sparse/hyper, SECOND multiplier
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is sparse/hyper
// multiply op is GxB_SECOND_* for any type, including user-defined

#define GB_AXB_SAXPY_GENERIC_METHOD GB_AxB_saxpy3_generic_second 
#define GB_GENERIC_C_IS_SPARSE_OR_HYPERSPARSE  1
#define GB_GENERIC_FLIPXY                      0
#define GB_GENERIC_NOFLIPXY                    0
#define GB_GENERIC_IDX_FLIPXY                  0
#define GB_GENERIC_IDX_NOFLIPXY                0
#define GB_GENERIC_OP_IS_FIRST                 0
#define GB_GENERIC_OP_IS_SECOND                1

#include "mxm/factory/GB_AxB_saxpy_generic_method.c"

