//------------------------------------------------------------------------------
// GB_wait_macros.h: determine if a matrix has pending tuples, zombies, ...
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_WAIT_MACROS_H
#define GB_WAIT_MACROS_H

// true if a matrix has pending tuples
#define GB_PENDING(A) ((A) != NULL && (A)->Pending != NULL)

// true if a matrix is allowed to have pending tuples
#define GB_PENDING_OK(A) (GB_PENDING (A) || !GB_PENDING (A))

// true if a matrix has zombies
#define GB_ZOMBIES(A) ((A) != NULL && (A)->nzombies > 0)

// true if a matrix is allowed to have zombies
#define GB_ZOMBIES_OK(A) (((A) == NULL) || ((A) != NULL && (A)->nzombies >= 0))

// true if a matrix has pending tuples or zombies
#define GB_PENDING_OR_ZOMBIES(A) (GB_PENDING (A) || GB_ZOMBIES (A))

// true if a matrix is jumbled
#define GB_JUMBLED(A) ((A) != NULL && (A)->jumbled)

// true if a matrix is allowed to be jumbled
#define GB_JUMBLED_OK(A) (GB_JUMBLED (A) || !GB_JUMBLED (A))

// true if a matrix has pending tuples, zombies, or is jumbled
#define GB_ANY_PENDING_WORK(A) \
    (GB_PENDING (A) || GB_ZOMBIES (A) || GB_JUMBLED (A))

#endif

