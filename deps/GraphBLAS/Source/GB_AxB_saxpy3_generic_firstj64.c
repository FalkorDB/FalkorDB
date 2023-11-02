//------------------------------------------------------------------------------
// GB_AxB_saxpy3_generic_firstj64.c: C=A*B, C sparse/hyper, FIRSTJ multiplier
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is sparse/hyper
// multiply op is GxB_FIRSTJ_INT64, GxB_FIRSTJ1_INT64,
// GxB_SECONDI_INT64, or GxB_SECONDI1_INT64

#define GB_AXB_SAXPY_GENERIC_METHOD GB_AxB_saxpy3_generic_firstj64 
#define GB_GENERIC_C_IS_SPARSE_OR_HYPERSPARSE  1
#define GB_GENERIC_OP_IS_POSITIONAL            1
#define GB_GENERIC_FLIPXY                      0
#define GB_GENERIC_OP_IS_INT64                 1
#define GB_GENERIC_OP_IS_FIRSTI                0
#define GB_GENERIC_OP_IS_FIRSTJ                1
#define GB_GENERIC_OP_IS_FIRST                 0
#define GB_GENERIC_OP_IS_SECOND                0

#include "GB_AxB_saxpy_generic_method.c"

