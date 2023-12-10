//------------------------------------------------------------------------------
// GB_index.h: definitions for index lists and types of assignments
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_INDEX_H
#define GB_INDEX_H

//------------------------------------------------------------------------------
// maximum matrix or vector dimension
//------------------------------------------------------------------------------

#ifndef GrB_INDEX_MAX
#define GrB_INDEX_MAX ((GrB_Index) (1ULL << 60) - 1)
#endif

#define GB_NMAX (GrB_INDEX_MAX + 1)

//------------------------------------------------------------------------------
// kind of index list, Ikind and Jkind, and assign variations
//------------------------------------------------------------------------------

#define GB_ALL 0
#define GB_RANGE 1
#define GB_STRIDE 2
#define GB_LIST 3

#define GB_ASSIGN 0
#define GB_SUBASSIGN 1
#define GB_ROW_ASSIGN 2
#define GB_COL_ASSIGN 3

#endif

