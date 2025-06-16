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



void _numInEntryFirst(uint64_t *z, const uint64_t *x, const uint64_t *y) {
	if(SCALAR_ENTRY(*x))
	{
		
		*z = (uint64_t) 1;
	}
	else
	{
		GrB_Vector v = AS_VECTOR(*x);
		GrB_Info info = GrB_Vector_nvals(z, v);
		ASSERT(info == GrB_SUCCESS);
	}
}

// Compute the in or out degree of each node in degree
//
// arguments:
// 'degree' input / output degree vector. Degree of node i is added to degree[i]
// 'dest' input vector. Boolean vector with entries to be counted for degree.
// 'T' Tensor being used to find the degree.
// 'ops' input. DEG_[OUT/IN]DEGREE: compute [out/in]degree. 
// 				DEG_TENSOR: compute tensor degree
// returns:
// GrB_SUCCESS on success otherwise a GraphBLAS error
GrB_Info TensorDegree  
(
	GrB_Vector degree,      // [input / output] degree vector with values where 
	                        // the degree should be accumulated
	const GrB_Vector dest,  // [input] possible destination / source nodes
	const Tensor T,         // matrix with tensor entries
	Degree_Options ops      // options:
	                        // DEG_OUTDEGREE: compute outdegree
	                        // DEG_INDEGREE: compute indegree
	                        // DEG_TENSOR: compute tensor degree
) {
	ASSERT(T      != NULL);
	ASSERT(degree != NULL);
	ASSERT(ops &  (DEG_INDEGREE | DEG_OUTDEGREE));
	GrB_Info info;

	if((ops & (DEG_INDEGREE | DEG_OUTDEGREE)) == (DEG_INDEGREE | DEG_OUTDEGREE) )
	{
		// Sum the in and out degrees
		info = TensorDegree(degree, dest, T, ops ^ DEG_INDEGREE);
		info |= TensorDegree(degree, dest, T, ops ^ DEG_OUTDEGREE);
		return info;
	}

	GrB_Matrix M  = Delta_Matrix_M(T);
	GrB_Matrix Dp = Delta_Matrix_Dp(T);
	GrB_Matrix Dm = Delta_Matrix_Dm(T);

	GrB_BinaryOp countEntry         = NULL; 
	GrB_Semiring plus_count_uint64  = NULL;
	GrB_Semiring semiring           = GxB_PLUS_PAIR_UINT64;
	GrB_Descriptor desc             = (ops & DEG_INDEGREE)? GrB_DESC_ST0: GrB_DESC_S;
	
	if(ops & DEG_TENSOR)
	{
		// create custom semiring
		// a * b = count entries in a
		// a + b = a + b
		info = GrB_BinaryOp_new(
			&countEntry, (GxB_binary_function) _numInEntryFirst, 
			GrB_UINT64, GrB_UINT64, GrB_UINT64);
		ASSERT(info == GrB_SUCCESS);
		info = GrB_Semiring_new(&plus_count_uint64, GrB_PLUS_MONOID_UINT64,
				countEntry);
		ASSERT(info == GrB_SUCCESS);

		semiring = plus_count_uint64;
	}

	//--------------------------------------------------------------------------
	// compute the degree
	//--------------------------------------------------------------------------
	// GraphBLAS decides wether to explicitly transpose.
	info = GrB_mxv(degree, degree, GrB_PLUS_UINT64, semiring, M, dest, desc);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_mxv(
		degree, degree, GrB_PLUS_UINT64, semiring, Dp, dest, desc);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_mxv(
		degree, degree, GrB_MINUS_UINT64, GxB_PLUS_PAIR_UINT64, Dm, dest, desc);
	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// clean up
	//--------------------------------------------------------------------------

	info = GrB_free(&countEntry);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_free(&plus_count_uint64);
	ASSERT(info == GrB_SUCCESS);
	return info;
}
