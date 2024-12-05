//------------------------------------------------------------------------------
// GB_iceil.h: definitions for ceiling (a/b)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_ICEIL_H
#define GB_ICEIL_H

// ceiling of a/b for two integers a and b
#define GB_ICEIL(a,b) (((a) + (b) - 1) / (b))

#endif

