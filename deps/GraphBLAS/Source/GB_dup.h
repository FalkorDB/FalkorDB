//------------------------------------------------------------------------------
// GB_dup.h: definitions for GB_dup*
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_DUP_H
#define GB_DUP_H

GrB_Info GB_dup             // make an exact copy of a matrix
(
    GrB_Matrix *Chandle,    // handle of output matrix to create
    const GrB_Matrix A,     // input matrix to copy
    GB_Werk Werk
) ;

GrB_Info GB_dup_worker      // make an exact copy of a matrix
(
    GrB_Matrix *Chandle,    // output matrix, NULL or existing static/dynamic
    const bool C_iso,       // if true, construct C as iso
    const GrB_Matrix A,     // input matrix to copy
    const bool numeric,     // if true, duplicate the numeric values; if A is
                            // iso, only the first entry is copied, regardless
                            // of C_iso on input
    const GrB_Type ctype    // type of C, if numeric is false
) ;

#endif

