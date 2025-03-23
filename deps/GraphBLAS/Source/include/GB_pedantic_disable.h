//------------------------------------------------------------------------------
// GB_pedantic_disable.h: disable -Wpedantic
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#if GB_COMPILER_GCC
#pragma GCC diagnostic ignored "-Wpedantic"
// #pragma GCC diagnostic ignored "-pedantic"
#endif

