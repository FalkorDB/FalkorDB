//------------------------------------------------------------------------------
// GB_AxB_saxpy3_generic_first.c: C=A*B, C sparse/hyper, FIRST multiplier
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is sparse/hyper
// multiply op is GxB_FIRST_* for any type, including user-defined

#define GB_AXB_SAXPY_GENERIC_METHOD GB_AxB_saxpy3_generic_first 
#define GB_GENERIC_C_IS_SPARSE_OR_HYPERSPARSE  1
#define GB_GENERIC_OP_IS_POSITIONAL            0
#define GB_GENERIC_FLIPXY                      0
#define GB_GENERIC_OP_IS_INT64                 0
#define GB_GENERIC_OP_IS_FIRSTI                0
#define GB_GENERIC_OP_IS_FIRSTJ                0
#define GB_GENERIC_OP_IS_FIRST                 1
#define GB_GENERIC_OP_IS_SECOND                0

#include "GB_AxB_saxpy_generic_method.c"

