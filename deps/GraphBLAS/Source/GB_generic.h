//------------------------------------------------------------------------------
// GB_generic.h: definitions for all generic methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This header is #include'd just before using any templates for generic
// methods, which use memcpy for assignments, and function pointers for any
// operators.

// tell any template that it's generic
#define GB_GENERIC

// generic methods do not allow for any simd vectorization
#undef  GB_PRAGMA_SIMD_REDUCTION_MONOID
#define GB_PRAGMA_SIMD_REDUCTION_MONOID(z)

#undef  GB_PRAGMA_SIMD_VECTORIZE
#define GB_PRAGMA_SIMD_VECTORIZE

// all data types are GB_void
#undef  GB_A_TYPE
#define GB_A_TYPE GB_void

#undef  GB_A2TYPE
#define GB_A2TYPE GB_void

#undef  GB_B_TYPE
#define GB_B_TYPE GB_void

#undef  GB_B2TYPE
#define GB_B2TYPE GB_void

#undef  GB_Z_TYPE
#define GB_Z_TYPE GB_void

#undef  GB_C_TYPE
#define GB_C_TYPE GB_void

