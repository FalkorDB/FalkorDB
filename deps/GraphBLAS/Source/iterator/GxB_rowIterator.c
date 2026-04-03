//------------------------------------------------------------------------------
// GxB_rowIterator_*: iterate over the rows of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

#undef GxB_rowIterator_attach
#undef GxB_rowIterator_kount
#undef GxB_rowIterator_seekRow
#undef GxB_rowIterator_kseek
#undef GxB_rowIterator_nextRow
#undef GxB_rowIterator_nextCol
#undef GxB_rowIterator_getRowIndex
#undef GxB_rowIterator_getColIndex

#if !defined ( GBMATLAB )

// These methods are not tested by the GraphBLAS/Test or GraphBLAS/Tcov
// test coverage suite, because the conflict with libmwgraphblas.so inside
// MATLAB.

//------------------------------------------------------------------------------
// GxB_rowIterator_attach: attach a row iterator to a matrix
//------------------------------------------------------------------------------

// On input, the iterator must already exist, having been created by
// GxB_Iterator_new.

// GxB_rowIterator_attach attaches a row iterator to a matrix.  If the iterator
// is already attached to a matrix, it is detached and then attached to the
// given matrix A.

// The following error conditions are returned:
// GrB_NULL_POINTER:    if the iterator or A are NULL.
// GrB_INVALID_OBJECT:  if the matrix A is invalid.
// GrB_NOT_IMPLEMENTED: if the matrix A cannot be iterated by row.
// GrB_OUT_OF_MEMORY:   if the method runs out of memory.

// If successful, the row iterator is attached to the matrix, but not to any
// specific row.  Use GxB_rowIterator_*seek* to move the iterator to a row.

GrB_Info GxB_rowIterator_attach
(
    GxB_Iterator iterator,
    GrB_Matrix A,
    GrB_Descriptor desc
)
{
    return (GB(Iterator_attach)(iterator, A, GxB_BY_ROW, desc)) ;
}

//------------------------------------------------------------------------------
// GxB_rowIterator_kount: upper bound on the # of nonempty rows of a matrix
//------------------------------------------------------------------------------

// On input, the row iterator must be attached to a matrix, but need not be at
// any specific row; results are undefined if this condition is not met.

// GxB_rowIterator_kount returns an upper bound on the # of non-empty rows of a
// matrix.  A GraphBLAS library may always return this as simply nrows(A), but
// in some libraries, it may be a value between the # of rows with at least one
// entry, and nrows(A), inclusive.  Any value in this range is a valid return
// value from this function.

// For SuiteSparse:GraphBLAS: If A is m-by-n, and sparse, bitmap, or full, then
// kount == m.  If A is hypersparse, kount is the # of vectors held in the data
// structure for the matrix, some of which may be empty, and kount <= m.

uint64_t GxB_rowIterator_kount (GxB_Iterator iterator)
{
    return (iterator->anvec) ;
}

//------------------------------------------------------------------------------
// GxB_rowIterator_seekRow:  move a row iterator to a different row of a matrix
//------------------------------------------------------------------------------

// On input, the row iterator must be attached to a matrix, but need not be at
// any specific row; results are undefined if this condition is not met.

// GxB_rowIterator_seekRow moves a row iterator to the first entry of A(row,:).
// If A(row,:) has no entries, the iterator may move to the first entry of next
// nonempty row i for some i > row.  The row index can be determined by
// GxB_rowIterator_getRowIndex.

// For SuiteSparse:GraphBLAS: If the matrix is hypersparse, and the row
// does not appear in the hyperlist, then the iterator is moved to the first
// row after the given row that does appear in the hyperlist.

// The method is always successful; the following are conditions are returned:
// GxB_EXHAUSTED:   if the row index is >= nrows(A); the row iterator is
//                  exhausted, but is still attached to the matrix.
// GrB_NO_VALUE:    if the row index is valid but A(row,:) has no entries; the
//                  row iterator is positioned at A(row,:).
// GrB_SUCCESS:     if the row index is valid and A(row,:) has at least one
//                  entry. The row iterator is positioned at A(row,:).
//                  GxB_rowIterator_get* can be used to return the indices of
//                  the first entry in A(row,:), and GxB_Iterator_get* can
//                  return its value.

GrB_Info GxB_rowIterator_seekRow (GxB_Iterator iterator, uint64_t row)
{
    return (GB(Iterator_rc_seek)(iterator, row, false)) ;
}

//------------------------------------------------------------------------------
// GxB_rowIterator_kseek:  move a row iterator to a different row of a matrix
//------------------------------------------------------------------------------

// On input, the row iterator must be attached to a matrix, but need not be at
// any specific row; results are undefined if this condition is not met.

// GxB_rowIterator_kseek is identical to GxB_rowIterator_seekRow, except for
// how the row index is specified.  The row is the kth non-empty row of A.
// More precisely, k is in the range 0 to kount-1, where kount is the value
// returned by GxB_rowIterator_kount.

GrB_Info GxB_rowIterator_kseek (GxB_Iterator iterator, uint64_t k)
{
    return (GB(Iterator_rc_seek)(iterator, k, true)) ;
}

//------------------------------------------------------------------------------
// GxB_rowIterator_nextRow: move a row iterator to the next row of a matrix
//------------------------------------------------------------------------------

// On input, the row iterator must already be attached to a matrix via a prior
// call to GxB_rowIterator_attach, and the iterator must be at a specific row,
// via a prior call to GxB_rowIterator_*seek* or GxB_rowIterator_nextRow;
// results are undefined if this condition is not met.

// If the the row iterator is currently at A(row,:), it is moved to A(row+1,:),
// or to the first non-empty row after A(row,:), at the discretion of this
// method.  That is, empty rows may be skipped.

// The method is always successful, and the return conditions are identical to
// the return conditions of GxB_rowIterator_seekRow.

GrB_Info GxB_rowIterator_nextRow (GxB_Iterator iterator)
{
    return (GB_Iterator_rc_knext (iterator)) ;
}

//------------------------------------------------------------------------------
// GxB_rowIterator_nextCol: move a row iterator to the next entry in A(row,:)
//------------------------------------------------------------------------------

// On input, the row iterator must already be attached to a matrix via a prior
// call to GxB_rowIterator_attach, and the iterator must be at a specific row,
// via a prior call to GxB_rowIterator_*seek* or GxB_rowIterator_nextRow;
// results are undefined if this condition is not met.

// The method is always successful, and returns the following conditions:
// GrB_NO_VALUE:    If the iterator is already exhausted, or if there is no
//                  entry in the current A(row,;),
// GrB_SUCCESS:     If the row iterator has been moved to the next entry in
//                  A(row,:).

GrB_Info GxB_rowIterator_nextCol (GxB_Iterator iterator)
{
    return (GB_Iterator_rc_inext (iterator)) ;
}

//------------------------------------------------------------------------------
// GxB_rowIterator_getRowIndex: get current row index of a row iterator
//------------------------------------------------------------------------------

// On input, the iterator must be already successfully attached to matrix as a
// row iterator; results are undefined if this condition is not met.

// The method returns nrows(A) if the iterator is exhausted, or the current
// row index otherwise.  There need not be any entry in the current row.
// Zero is returned if the iterator is attached to the matrix but
// GxB_rowIterator_*seek* has not been called, but this does not mean the
// iterator is positioned at row zero.

uint64_t GxB_rowIterator_getRowIndex (GxB_Iterator iterator)
{
    return (GB_Iterator_rc_getj (iterator)) ;
}

//------------------------------------------------------------------------------
// GxB_rowIterator_getColIndex: get current column index of a row iterator
//------------------------------------------------------------------------------

// On input, the iterator must be already successfully attached to matrix as a
// row iterator, and in addition, the row iterator must be positioned at a
// valid entry present in the matrix.  That is, the last call to
// GxB_rowIterator_*seek* or GxB_rowIterator_*next*, must have returned
// GrB_SUCCESS.  Results are undefined if this condition is not met.

uint64_t GxB_rowIterator_getColIndex (GxB_Iterator iterator)
{
    return (GB_Iterator_rc_geti (iterator)) ;
}

#endif

