//------------------------------------------------------------------------------
// GB_select_shared_definitions.h: common macros for select kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_select_shared_definitions.h provides default definitions for all select
// kernels, if the special cases have not been #define'd prior to #include'ing
// this file.  This file is shared by generic, factory, and both CPU and
// CUDA JIT kernels.

#include "GB_kernel_shared_definitions.h"

#ifndef GB_SELECT_SHARED_DEFINITIONS_H
#define GB_SELECT_SHARED_DEFINITIONS_H

// 1 if C is iso
#ifndef GB_ISO_SELECT
#define GB_ISO_SELECT 0
#endif

#endif

