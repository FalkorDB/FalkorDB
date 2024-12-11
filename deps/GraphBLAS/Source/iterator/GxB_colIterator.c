//------------------------------------------------------------------------------
// GxB_colIterator_*: iterate over columns of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

#undef GxB_colIterator_attach
#undef GxB_colIterator_kount
#undef GxB_colIterator_seekCol
#undef GxB_colIterator_kseek
#undef GxB_colIterator_nextCol
#undef GxB_colIterator_nextRow
#undef GxB_colIterator_getColIndex
#undef GxB_colIterator_getRowIndex

//------------------------------------------------------------------------------

// The column iterator is analoguous to the row iterator.

GrB_Info GxB_colIterator_attach
(
    GxB_Iterator iterator,
    GrB_Matrix A,
    GrB_Descriptor desc
)
{ 
    // attach a column iterator to a matrix
    return (GB_Iterator_attach (iterator, A, GxB_BY_COL, desc)) ;
}

GrB_Index GxB_colIterator_kount (GxB_Iterator iterator)
{ 
    // return # of nonempty columns of the matrix
    return (iterator->anvec) ;
}

GrB_Info GxB_colIterator_seekCol (GxB_Iterator iterator, GrB_Index col)
{ 
    // move a column iterator that is already attached to A, to the first
    // entry of A(:,col)
    return (GB_Iterator_rc_seek (iterator, col, false)) ;
}

GrB_Info GxB_colIterator_kseek (GxB_Iterator iterator, GrB_Index k)
{ 
    // move a column iterator that is already attached to A, to the first
    // entry of the kth non-empty column of A
    return (GB_Iterator_rc_seek (iterator, k, true)) ;
}

GrB_Info GxB_colIterator_nextCol (GxB_Iterator iterator)
{ 
    // move a column iterator to the first entry of the next column
    return (GB_Iterator_rc_knext (iterator)) ;
}

GrB_Info GxB_colIterator_nextRow (GxB_Iterator iterator)
{ 
    // move a column iterator to the next row in the same column
    return (GB_Iterator_rc_inext (iterator)) ;
}

GrB_Index GxB_colIterator_getColIndex (GxB_Iterator iterator)
{ 
    // return the column index of the current entry for a column iterator
    return (GB_Iterator_rc_getj (iterator)) ;
}

GrB_Index GxB_colIterator_getRowIndex (GxB_Iterator iterator)
{ 
    // return the row index of the current entry for a column iterator
    return (GB_Iterator_rc_geti (iterator)) ;
}

