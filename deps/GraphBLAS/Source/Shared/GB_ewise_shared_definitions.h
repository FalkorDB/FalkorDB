//------------------------------------------------------------------------------
// GB_ewise_shared_definitions.h: common macros for ewise kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_ewise_shared_definitions.h provides default definitions for all ewise
// kernels, if the special cases have not been #define'd prior to #include'ing
// this file.  This file is shared by generic, factory, and both CPU and
// CUDA JIT kernels.

#include "GB_kernel_shared_definitions.h"

#ifndef GB_EWISE_SHARED_DEFINITIONS_H
#define GB_EWISE_SHARED_DEFINITIONS_H

// C(i,j) = op (aij,bij) ;
#ifndef GB_EWISEOP
#define GB_EWISEOP(Cx,p,aij,bij,i,j) GB_BINOP (Cx [p], aij, bij, i, j)
#endif

// Cx [p] = z
#ifndef GB_PUTC
#define GB_PUTC(z,Cx,p) Cx [p] = z
#endif

// 1 if operator is second
#ifndef GB_OP_IS_SECOND
#define GB_OP_IS_SECOND 0
#endif

// copy A(i,j) to C(i,j)
#ifndef GB_COPY_A_to_C
#define GB_COPY_A_to_C(Cx,pC,Ax,pA,A_iso) Cx [pC] = Ax [(A_iso) ? 0 : (pA)]
#endif

// copy B(i,j) to C(i,j)
#ifndef GB_COPY_B_to_C
#define GB_COPY_B_to_C(Cx,pC,Bx,pB,B_iso) Cx [pC] = Bx [(B_iso) ? 0 : (pB)]
#endif

// 1 if C and A have the same type
#ifndef GB_CTYPE_IS_ATYPE
#define GB_CTYPE_IS_ATYPE 1
#endif

// 1 if C and B have the same type
#ifndef GB_CTYPE_IS_BTYPE
#define GB_CTYPE_IS_BTYPE 1
#endif

#endif

