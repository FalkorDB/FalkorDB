//------------------------------------------------------------------------------
// GB_prefix.h: GraphBLAS naming prefix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_PREFIX_H
#define GB_PREFIX_H

#define GB_CAT3(x,y,z) x ## y ## z
#define GB_CAT4(w,x,y,z) w ## x ## y ## z
#define GB_CAT5(v,w,x,y,z) v ## w ## x ## y ## z

#define GB_EVAL3(x,y,z) GB_CAT3 (x,y,z)
#define GB_EVAL4(w,x,y,z) GB_CAT4 (w,x,y,z)
#define GB_EVAL5(v,w,x,y,z) GB_CAT5 (v,w,x,y,z)

#endif

