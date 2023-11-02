//------------------------------------------------------------------------------
// GB_wait.h: definitions for GB_wait
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_WAIT_H
#define GB_WAIT_H

GrB_Info GB_block   // apply all pending computations if blocking mode enabled
(
    GrB_Matrix A,
    GB_Werk Werk
) ;

GrB_Info GB_wait                // finish all pending computations
(
    GrB_Matrix A,               // matrix with pending computations
    const char *name,           // name of the matrix
    GB_Werk Werk
) ;

GrB_Info GB_unjumble        // unjumble a matrix
(
    GrB_Matrix A,           // matrix to unjumble
    GB_Werk Werk
) ;

// wait if condition holds
#define GB_WAIT_IF(condition,A,name)                                    \
{                                                                       \
    if (condition)                                                      \
    {                                                                   \
        GrB_Info info ;                                                 \
        GB_OK (GB_wait ((GrB_Matrix) A, name, Werk)) ;                  \
    }                                                                   \
}

// do all pending work:  zombies, pending tuples, and unjumble
#define GB_MATRIX_WAIT(A) GB_WAIT_IF (GB_ANY_PENDING_WORK (A), A, GB_STR (A))

// do all pending work if pending tuples; zombies and jumbled are OK
#define GB_MATRIX_WAIT_IF_PENDING(A) GB_WAIT_IF (GB_PENDING (A), A, GB_STR (A))

// delete zombies and assemble any pending tuples; jumbled is O
#define GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES(A)                         \
    GB_WAIT_IF (GB_PENDING_OR_ZOMBIES (A), A, GB_STR (A))

// ensure A is not jumbled
#define GB_MATRIX_WAIT_IF_JUMBLED(A) GB_WAIT_IF (GB_JUMBLED (A), A, GB_STR (A))

#endif

