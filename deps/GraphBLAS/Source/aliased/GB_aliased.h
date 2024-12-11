//------------------------------------------------------------------------------
// GB_aliased.h: definitions for aliased and shallow matrices
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_ALIASED_H
#define GB_ALIASED_H

// GraphBLAS allows all inputs to all user-accessible objects to be aliased, as
// in GrB_mxm (C, C, accum, C, C, ...), which is valid.  Internal routines are
// more restrictive.

// GB_any_aliased also checks the content of A and B
bool GB_any_aliased             // determine if A and B are aliased in any way
(
    GrB_Matrix A,           // input A matrix
    GrB_Matrix B            // input B matrix
) ;

// GB_all_aliased is true if A and B are identical 
bool GB_all_aliased         // determine if A and B are all aliased
(
    GrB_Matrix A,           // input A matrix
    GrB_Matrix B            // input B matrix
) ;

// matrices returned to the user are never shallow; internal matrices may be
bool GB_is_shallow              // true if any component of A is shallow
(
    GrB_Matrix A                // matrix to query
) ;

#endif

