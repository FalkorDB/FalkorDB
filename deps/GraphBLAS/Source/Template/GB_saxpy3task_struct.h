//------------------------------------------------------------------------------
// GB_saxpy3task_struct.h: definitions for C=A*B saxpy3 method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_SAXPY3TASK_STRUCT_H
#define GB_SAXPY3TASK_STRUCT_H

// See GB_AxB_saxpy3.h for a description.

typedef struct
{
    int64_t start ;     // starting vector for coarse task, p for fine task
    int64_t end ;       // ending vector for coarse task, p for fine task
    int64_t vector ;    // -1 for coarse task, vector j for fine task
    int64_t hsize ;     // size of hash table
    int64_t *Hi ;       // Hi array for hash table (coarse hash tasks only)
    GB_void *Hf ;       // Hf array for hash table (int8_t or int64_t)
    GB_void *Hx ;       // Hx array for hash table
    int64_t my_cjnz ;   // # entries in C(:,j) found by this fine task
    int leader ;        // leader fine task for the vector C(:,j)
    int team_size ;     // # of fine tasks in the team for vector C(:,j)
}
GB_saxpy3task_struct ;

#endif

