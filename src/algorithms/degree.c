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

// Returns the degree vector of a tensor with no multi-edges
GrB_Info TensorDegree_flat
(
	GrB_Vector degree,  // [input / output] degree vector with values where 
						// the degree should be added
	GrB_Vector dest,    // [input] possible destination / source nodes
	Tensor T,           // matrix with tensor entries
    bool transpose      
) {
    GrB_Descriptor desc  = transpose? GrB_DESC_ST0: GrB_DESC_S;
    GrB_Info info;

    info = Delta_mxv_count(
		degree, degree, GrB_PLUS_UINT64, GxB_PLUS_PAIR_UINT64, T, dest, desc);
	return info;
}

static void _numInEntryFirst(uint64_t *z, const uint64_t *x, const uint64_t *y) {
	if(SCALAR_ENTRY(*x))
	{
		*z = (uint64_t) 1;
	}
	else
	{
		GrB_Vector v = AS_VECTOR(*x);
		if(v){
			GrB_Info info = GrB_Vector_nvals(z, v);
			ASSERT(info == GrB_SUCCESS);
		} else {
			*z = 0;
		}
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
	

	if((ops & (DEG_INDEGREE | DEG_OUTDEGREE)) == (DEG_INDEGREE | DEG_OUTDEGREE) ){
		// Sum the in and out degrees
		info = TensorDegree(degree, dest, T, ops ^ DEG_INDEGREE);
		ASSERT(info == GrB_SUCCESS);
		info = TensorDegree(degree, dest, T, ops ^ DEG_OUTDEGREE);
		ASSERT(info == GrB_SUCCESS);
		return info;
	}
	GrB_BinaryOp countEntry         = NULL; 
	GrB_Semiring plus_count_uint64  = NULL;
	GrB_Descriptor desc             = (ops & DEG_INDEGREE)? GrB_DESC_ST0: GrB_DESC_S;

	if((ops & DEG_TENSOR) == 0){
		return Delta_mxv_count(
			degree, degree, GrB_PLUS_UINT64, GxB_PLUS_PAIR_UINT64, T, dest, desc
		);
	}

	
	

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

	//--------------------------------------------------------------------------
	// compute the degree
	//--------------------------------------------------------------------------
	// GraphBLAS decides wether to explicitly transpose.
	info = Delta_mxv(
		degree, degree, GrB_PLUS_UINT64, plus_count_uint64, T, dest, desc);
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



// defining macros to be used by _sum_tensor_attributes functions.
// see comments in tensor_utils.h
#define FUNCTION_IDENTITY 0.0

#define GET_VALUE(z, x)                                                        \
    Edge edge;                                                                 \
	Graph_GetEdge(ctx->g, (x), &edge);                                         \
    SIValue *temp = GraphEntity_GetProperty(                                   \
        (GraphEntity *) &edge, ctx->attribute);                                \
    if(temp != ATTRIBUTE_NOTFOUND && (SI_NUMERIC & SI_TYPE(*temp))) {          \
        SIValue_ToDouble(temp, &(z));                                          \
    } else {                                                                   \
        (z) = FUNCTION_IDENTITY;                                               \
    }

#define ACCUM(z,x) ((z) += (x))

// Sum up all of the weights within an entry
static void _sum_tensor_attributes (
	double *z, 
	const uint64_t *x, 
	GrB_Index ix, 
	GrB_Index jx, 
	const bool *y, 
	GrB_Index iy, 
	GrB_Index jy, 
	const FDB_degree_ctx *ctx 
) TENSORPICK(double)

static void _flat_get_attribute (
	double *z, 
	const uint64_t *x, 
	GrB_Index ix, 
	GrB_Index jx, 
	const bool *y, 
	GrB_Index iy, 
	GrB_Index jy, 
	const FDB_degree_ctx *ctx 
){
	if(*x == MSB_MASK){
		*z = 0;
	} else {
		GET_VALUE(*z, *x);
	}
}
#undef FUNCTION_IDENTITY
#undef GET_VALUE
#undef ACCUM

// Compute the in or out degree of each node in degree
//
// arguments:
// 'degree' input / output degree vector. Degree of node i is added to degree[i]
//    degree is a double value, the sum of the weights of adjacent relationships
// 'dest' input vector. Boolean vector with entries to be counted for degree.
// 'T' Tensor being used to find the degree.
// 'ops' input. DEG_[OUT/IN]DEGREE: compute [out/in]degree. 
// 				DEG_TENSOR: compute tensor degree
// returns:
// GrB_SUCCESS on success otherwise a GraphBLAS error
GrB_Info TensorDegree_weighted
(
	GrB_Vector degree,  // [input / output] degree vector with values where 
	                    //         the degree should be added
	GrB_Vector dest,    // [input] possible destination / source nodes
	Tensor T,           // matrix with tensor entries
	Degree_Options ops,
	FDB_degree_ctx ctx
) {
	ASSERT (T             != NULL);
	ASSERT (degree        != NULL);
	ASSERT (ops &         (DEG_INDEGREE | DEG_OUTDEGREE));
	ASSERT (ctx.g         != NULL);
	ASSERT (ctx.attribute != ATTRIBUTE_ID_NONE);

	GrB_Info info;

	if((ops & (DEG_INDEGREE | DEG_OUTDEGREE)) == (DEG_INDEGREE | DEG_OUTDEGREE) ){
		// Sum the in and out degrees
		info = TensorDegree_weighted(degree, dest, T, ops ^ DEG_INDEGREE, ctx) ;
		info |= TensorDegree_weighted(
			degree, dest, T, ops ^ DEG_OUTDEGREE, ctx) ;
		return info;
	}

	GxB_IndexBinaryOp getWeight_idxop   = NULL; 
	GrB_BinaryOp      getWeight         = NULL; 
	GrB_Semiring      plus_weight_fp64  = NULL;
	GrB_Descriptor    desc              = (ops & DEG_INDEGREE)? 
	                                        GrB_DESC_ST0: GrB_DESC_S;
	GrB_Type          deg_contx         = NULL;
	GrB_Scalar        ctx_scalar        = NULL;

	// create custom semiring
	// a * b = sum entries in a
	// a + b = a + b
	info = GrB_Type_new(&deg_contx, sizeof(FDB_degree_ctx)) ;
	info = GrB_Scalar_new(&ctx_scalar, deg_contx) ;
	ASSERT(info == GrB_SUCCESS);
	info = GrB_Scalar_setElement_UDT(ctx_scalar, (void *)&ctx) ;
	ASSERT(info == GrB_SUCCESS);
	info = GxB_IndexBinaryOp_new(
		&getWeight_idxop, (GxB_index_binary_function) 
		((ops & DEG_TENSOR)? _sum_tensor_attributes: _flat_get_attribute),
		GrB_FP64, GrB_UINT64, GrB_UINT64, deg_contx, NULL, NULL);
	ASSERT(info == GrB_SUCCESS);
	info = GxB_BinaryOp_new_IndexOp(&getWeight, getWeight_idxop, ctx_scalar) ;
	ASSERT(info == GrB_SUCCESS);
	info = GrB_Semiring_new(&plus_weight_fp64, GrB_PLUS_MONOID_FP64, 
		getWeight);
	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// compute the degree
	//--------------------------------------------------------------------------
	// GraphBLAS decides wether to explicitly transpose.
	info = Delta_mxv(
		degree, degree, GrB_PLUS_FP64, plus_weight_fp64, T, dest, desc);
	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// clean up
	//--------------------------------------------------------------------------
	info = GrB_free(&deg_contx);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_free(&ctx_scalar);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_free(&getWeight);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_free(&plus_weight_fp64);
	ASSERT(info == GrB_SUCCESS);
	return info;
}

