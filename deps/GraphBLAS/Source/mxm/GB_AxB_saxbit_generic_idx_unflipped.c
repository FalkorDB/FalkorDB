//------------------------------------------------------------------------------
// GB_AxB_saxbit_generic_idx_unflipped.c: C=A*B, C bitmap/full, unflipped idx
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap only.
// multiply op is index unflipped

#define GB_AXB_SAXPY_GENERIC_METHOD GB_AxB_saxbit_generic_idx_unflipped 
#define GB_GENERIC_C_IS_SPARSE_OR_HYPERSPARSE  0
#define GB_GENERIC_FLIPXY                      0
#define GB_GENERIC_NOFLIPXY                    0
#define GB_GENERIC_IDX_FLIPXY                  0
#define GB_GENERIC_IDX_NOFLIPXY                1
#define GB_GENERIC_OP_IS_FIRST                 0
#define GB_GENERIC_OP_IS_SECOND                0

#include "mxm/factory/GB_AxB_saxpy_generic_method.c"

