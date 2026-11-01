//------------------------------------------------------------------------------
// GB_add_shared_definitions: C(i,j) = A(i,j) + B(i,j) for eWiseAdd, eWiseUnion
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GB_LOAD_A and GB_LOAD_B: load a single entry of A(i,j) and B(i,j)
//------------------------------------------------------------------------------

#define GB_LOAD_A(aij, Ax,pA,A_iso) \
    GB_DECLAREA (aij) ;             \
    GB_GETA (aij, Ax,pA,A_iso)

#define GB_LOAD_B(bij, Bx,pB,B_iso) \
    GB_DECLAREB (bij) ;             \
    GB_GETB (bij, Bx,pB,B_iso)

//------------------------------------------------------------------------------
// GB_ADD_*_PLUS_*: compute C(i,j) = A(i,j) + B(i,j), etc
//------------------------------------------------------------------------------

#ifndef GB_C_ISO
#define GB_C_ISO 0
#endif

#ifndef GB_IS_EWISEUNION
#define GB_IS_EWISEUNION 0
#endif

#if GB_C_ISO

    // C is iso-valued: no numerical computations to compute C(i,j)
    #define GB_ADD_AIJ_PLUS_BETA(Cx,pC,Ax,pA,A_iso,beta,i,j)
    #define GB_ADD_ALPHA_PLUS_BIJ(Cx,pC,alpha,Bx,pB,B_iso,i,j)
    #define GB_ADD_AIJ_PLUS_BIJ(Cx,pC,Ax,pA,A_iso,Bx,pB,B_iso,i,j)

#else

    #define GB_ADD_AIJ_PLUS_BIJ(Cx,pC,Ax,pA,A_iso,Bx,pB,B_iso,i,j)  \
    {                                                               \
        /* C(i,j) = A(i,j) + B(i,j) */                              \
        GB_LOAD_A (aij, Ax, pA, A_iso) ;                            \
        GB_LOAD_B (bij, Bx, pB, B_iso) ;                            \
        GB_EWISEOP (Cx, pC, aij, bij, i, j) ;                       \
    }

    #if GB_IS_EWISEUNION

        #define GB_ADD_AIJ_PLUS_BETA(Cx,pC,Ax,pA,A_iso,beta,i,j)    \
        {                                                           \
            /* C (i,j) = A(i,j) + beta */                           \
            GB_LOAD_A (aij, Ax, pA, A_iso) ;                        \
            GB_EWISEOP (Cx, pC, aij, beta, i, j) ;                  \
        }

        #define GB_ADD_ALPHA_PLUS_BIJ(Cx,pC,alpha,Bx,pB,B_iso,i,j)  \
        {                                                           \
            /* C (i,j) = alpha + B(i,j) */                          \
            GB_LOAD_B (bij, Bx, pB, B_iso) ;                        \
            GB_EWISEOP (Cx, pC, alpha, bij, i, j) ;                 \
        }

    #else

        #define GB_ADD_AIJ_PLUS_BETA(Cx,pC,Ax,pA,A_iso,beta,i,j)    \
        {                                                           \
            /* C (i,j) = A (i,j) */                                 \
            GB_COPY_A_to_C (Cx, pC, Ax, pA, A_iso) ;                \
        }

        #define GB_ADD_ALPHA_PLUS_BIJ(Cx,pC,alpha,Bx,pB,B_iso,i,j)  \
        {                                                           \
            /* C (i,j) = B (i,j) */                                 \
            GB_COPY_B_to_C (Cx, pC, Bx, pB, B_iso) ;                \
        }

    #endif

#endif

