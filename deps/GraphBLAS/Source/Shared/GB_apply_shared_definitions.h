//------------------------------------------------------------------------------
// GB_apply_shared_definitions.h: common macros for unop kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_apply_shared_definitions.h provides default definitions for all unop
// kernels, if the special cases have not been #define'd prior to #include'ing
// this file.  This file is shared by generic, factory, and both CPU and
// CUDA JIT kernels.

#include "GB_kernel_shared_definitions.h"

#ifndef GB_APPLY_SHARED_DEFINITIONS_H
#define GB_APPLY_SHARED_DEFINITIONS_H

// nothing here so far that is unique to apply methods ...

#endif

