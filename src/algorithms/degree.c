// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

// LAGraph_Cached_OutDegree computes G->out_degree, where G->out_degree(i) is
// the number of entries in G->A (i,:).  If there are no entries in G->A (i,:),
// G->rowdgree(i) is not present in the structure of G->out_degree.  That is,
// G->out_degree contains no explicit zero entries.

#include "RG.h"
#include "degree.h"

// compute in/out degree for all nodes
//
// arguments:
// 'degree' output degree vector degree[i] contains the degree of node i
// 'A' adjacency matrix
//
// returns:
// GrB_SUCCESS on success otherwise a GraphBLAS error
GrB_Info Degree
(
	GrB_Vector *degree,  // [output] degree vector
	GrB_Matrix A         // graph matrix
) {
	ASSERT(A      != NULL);
	ASSERT(degree != NULL);

	GrB_Info info;
	*degree = NULL;

	GrB_Vector x                 = NULL;
	GrB_Vector _degree           = NULL;
	GrB_Semiring plus_one_uint64 = NULL;

	// create custome semiring
	// a * b = 1
	// a + b = a + b
	info = GrB_Semiring_new(&plus_one_uint64, GrB_PLUS_MONOID_UINT64,
			GrB_ONEB_UINT64);
	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// determine the size of adj
	//--------------------------------------------------------------------------

	GrB_Index nrows;
	GrB_Index ncols;

	info = GrB_Matrix_nrows(&nrows, A);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_ncols(&ncols, A);
	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// compute the degree
	//--------------------------------------------------------------------------

	info = GrB_Vector_new(&_degree, GrB_UINT64, nrows);
	ASSERT(info == GrB_SUCCESS);

	// x = zeros (ncols, 1)
	info = GrB_Vector_new(&x, GrB_UINT64, ncols);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_assign(x, NULL, NULL, 0, GrB_ALL, ncols, NULL);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_mxv(_degree, NULL, NULL, plus_one_uint64, A, x, NULL);
	ASSERT(info == GrB_SUCCESS);

	// flush matrix
	info = GrB_wait(_degree, GrB_MATERIALIZE);
	ASSERT(info == GrB_SUCCESS);

	// set output
	*degree = _degree;

	//--------------------------------------------------------------------------
	// clean up
	//--------------------------------------------------------------------------

	info = GrB_free(&x);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_free(&plus_one_uint64);
	ASSERT(info == GrB_SUCCESS);

	return info;
}

