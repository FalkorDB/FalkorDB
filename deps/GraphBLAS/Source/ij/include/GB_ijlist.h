//------------------------------------------------------------------------------
// GB_ijlist.h: return kth item, i = I [k], in an index list
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_IJLIST_H
#define GB_IJLIST_H

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

//------------------------------------------------------------------------------
// GB_IJLIST: given k, return the kth item i = I [k] in the list (32/64 bit)
//------------------------------------------------------------------------------

#define GB_IJLIST(I,k,Ikind,Icolon)                                         \
(                                                                           \
    (Ikind == GB_ALL) ?     (k) :                                           \
    ((Ikind == GB_RANGE) ?  (Icolon [GxB_BEGIN] + (k)) :                    \
    ((Ikind == GB_STRIDE) ? (Icolon [GxB_BEGIN] + (k) * Icolon [GxB_INC]) : \
    /* else GB_LIST */      (GB_IGET (I, k))))                              \
)

#endif

