//------------------------------------------------------------------------------
// GB_ewise_shared_definitions.h: common macros for ewise kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_ewise_shared_definitions.h provides default definitions for all ewise
// kernels, if the special cases have not been #define'd prior to #include'ing
// this file.  This file is shared by generic, factory, and both CPU and
// CUDA JIT kernels.

#include "include/GB_kernel_shared_definitions.h"

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

// 1 if C is iso-valued
#ifndef GB_C_ISO
#define GB_C_ISO 0
#endif

// 1 for eWiseUnion, 0 otherwise
#ifndef GB_IS_EWISEUNION
#define GB_IS_EWISEUNION 0
#endif

// 1 if C=method(A,B) computes the set union (eWiseAdd, eWiseUnion)
#ifndef GB_SET_UNION
#define GB_SET_UNION 0
#endif

// 1 if C=method(A,B) computes the set intersection (eWiseMult, C<M>=A)
#ifndef GB_SET_INTERSECTION
#define GB_SET_INTERSECTION 0
#endif

// 1 if C=method(A,B) computes the set difference (C<!M>=A)
#ifndef GB_SET_DIFFERENCE
#define GB_SET_DIFFERENCE 0
#endif

// 1 for C<M>=A or C<!M>=A where M is structural and passed in as the B matrix
#ifndef GB_IS_MASK_ASSIGN
#define GB_IS_MASK_ASSIGN 0
#endif

//------------------------------------------------------------------------------
// GB_LOAD_A and GB_LOAD_B: load a single entry of A(i,j) and B(i,j)
//------------------------------------------------------------------------------

#define GB_LOAD_A(aij, Ax,pA,A_iso) \
    GB_DECLAREA (aij) ;             \
    GB_GETA (aij, Ax,pA,A_iso)

#if GB_IS_MASK_ASSIGN
    // the values B are not used; it is the structural mask matrix M
    #define GB_LOAD_B(bij, Bx,pB,B_iso)
#else
    #define GB_LOAD_B(bij, Bx,pB,B_iso) \
        GB_DECLAREB (bij) ;             \
        GB_GETB (bij, Bx,pB,B_iso)
#endif

//------------------------------------------------------------------------------
// GB_EWISE_*_OP_*: compute C(i,j) = A(i,j) + B(i,j), etc
//------------------------------------------------------------------------------

#if GB_C_ISO

    // C is iso-valued: no numerical computations to compute C(i,j)
    #define GB_EWISE_AIJ_OP_BETA(Cx,pC,Ax,pA,A_iso,beta,i,j)
    #define GB_EWISE_ALPHA_OP_BIJ(Cx,pC,alpha,Bx,pB,B_iso,i,j)
    #define GB_EWISE_AIJ_OP_BIJ(Cx,pC,Ax,pA,A_iso,Bx,pB,B_iso,i,j)

#else

    #define GB_EWISE_AIJ_OP_BIJ(Cx,pC,Ax,pA,A_iso,Bx,pB,B_iso,i,j)  \
    {                                                               \
        /* C(i,j) = A(i,j) + B(i,j) */                              \
        GB_LOAD_A (aij, Ax, pA, A_iso) ;                            \
        GB_LOAD_B (bij, Bx, pB, B_iso) ;                            \
        GB_EWISEOP (Cx, pC, aij, bij, i, j) ;                       \
    }

    #if GB_IS_EWISEUNION

        #define GB_EWISE_AIJ_OP_BETA(Cx,pC,Ax,pA,A_iso,beta,i,j)    \
        {                                                           \
            /* C (i,j) = A(i,j) + beta */                           \
            GB_LOAD_A (aij, Ax, pA, A_iso) ;                        \
            GB_EWISEOP (Cx, pC, aij, beta, i, j) ;                  \
        }

        #define GB_EWISE_ALPHA_OP_BIJ(Cx,pC,alpha,Bx,pB,B_iso,i,j)  \
        {                                                           \
            /* C (i,j) = alpha + B(i,j) */                          \
            GB_LOAD_B (bij, Bx, pB, B_iso) ;                        \
            GB_EWISEOP (Cx, pC, alpha, bij, i, j) ;                 \
        }

    #else

        #define GB_EWISE_AIJ_OP_BETA(Cx,pC,Ax,pA,A_iso,beta,i,j)    \
        {                                                           \
            /* C (i,j) = A (i,j) */                                 \
            GB_COPY_A_to_C (Cx, pC, Ax, pA, A_iso) ;                \
        }

        #define GB_EWISE_ALPHA_OP_BIJ(Cx,pC,alpha,Bx,pB,B_iso,i,j)  \
        {                                                           \
            /* C (i,j) = B (i,j) */                                 \
            GB_COPY_B_to_C (Cx, pC, Bx, pB, B_iso) ;                \
        }

    #endif

#endif

#endif

