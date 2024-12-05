//------------------------------------------------------------------------------
// GB_Semiring_new.h: definitions for GB_Semiring_new
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_SEMIRING_NEW_H
#define GB_SEMIRING_NEW_H

GrB_Info GB_Semiring_new            // create a semiring
(
    GrB_Semiring semiring,          // semiring to create
    GrB_Monoid add,                 // additive monoid of the semiring
    GrB_BinaryOp multiply           // multiply operator of the semiring
) ;

#endif

