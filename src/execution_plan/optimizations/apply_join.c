/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "../../util/arr.h"
#include "../ops/op_filter.h"
#include "../ops/op_value_hash_join.h"
#include "../../util/rax_extensions.h"
#include "../ops/op_cartesian_product.h"
#include "../execution_plan_build/execution_plan_util.h"
#include "../execution_plan_build/execution_plan_modify.h"

#define NOT_RESOLVED -1

// applyJoin will try to locate situations where two disjoint
// streams can be joined on a key attribute, in which case the
// runtime complaxity is reduced from O(n^2) to O(nlogn + 2n)
// consider MATCH (a), (b) where a.v = b.v RETURN a,b
// prior to this optimization a and b will be combined via a
// cartesian product O(n^2) because a and b are related
// we require a.v = b.v, v acts as a join key in which case
// replacing the cartesian product by a join operation will
// 1. consume N additional memory
// 2. reduce the overall runtime by a factor of magnitude

// to be used as a possible output of _relate_exp_to_stream

// given an expression node from a filter tree, returns the stream number
// that fully resolves the expression's references
// exp: - Filter tree expression node
// stream_entities - streams to search the expressions referenced entities
// stream_count - amount of stream to search in (Left-to-Right)
// returns stream index if found NOT_RESOLVED
// if non of the stream resolve the expression
static int _relate_exp_to_stream
(
	AR_ExpNode *exp,
	rax **stream_entities,
	int stream_count
) {
	// collect the referenced entities in the expression
	rax *entities = raxNew();
	AR_EXP_CollectEntities(exp, entities);

	int stream_num;
	for(stream_num = 0; stream_num < stream_count; stream_num ++) {
		// see if the stream resolves all of the references
		if(raxIsSubset(stream_entities[stream_num], entities)) break;
	}
	raxFree(entities);

	// no single stream resolves all references
	if(stream_num == stream_count) return NOT_RESOLVED;
	return stream_num;
}

// tests to see if given filter can act as a join condition
static inline bool _applicableFilter
(
	const FT_FilterNode *f
) {
	return (f->t == FT_N_PRED && f->pred.op == OP_EQUAL);
}

// collects all consecutive filters beneath given op
static OpFilter **_locate_filters
(
	OpBase *cp
) {
	OpBase *parent = cp->parent;
	OpFilter **filters = array_new(OpFilter *, 0);

	while(parent && parent->type == OPType_FILTER) {
		OpFilter *filter_op = (OpFilter *)parent;

		if(_applicableFilter(filter_op->filterTree)) {
			array_append(filters, filter_op);
		}

		parent = parent->parent;
	}

	return filters;
}

// this function builds a Hash Join operation given its left and right
// branches and join criteria
static OpBase *_build_hash_join_op
(
	const ExecutionPlan *plan,
	OpBase *left_branch,
	OpBase *right_branch,
	AR_ExpNode *lhs_join_exp,
	AR_ExpNode *rhs_join_exp
) {
	OpBase *value_hash_join;

	// the Value Hash Join will cache its left-hand stream
	// to reduce the cache size prefer to cache the stream which will produce
	// the smallest number of records
	// our current heuristic for this is to prefer a stream which contains a
	// filter operation
	bool left_branch_filtered =
		(ExecutionPlan_LocateOp(left_branch, OPType_FILTER) != NULL);
	bool right_branch_filtered =
		(ExecutionPlan_LocateOp(right_branch, OPType_FILTER) != NULL);

	if(!left_branch_filtered && right_branch_filtered) {
		// only the RHS stream is filtered
		// swap the input streams and expressions
		value_hash_join = NewValueHashJoin(plan, rhs_join_exp, lhs_join_exp);
		OpBase *t = left_branch;
		left_branch = right_branch;
		right_branch = t;
	} else {
		value_hash_join = NewValueHashJoin(plan, lhs_join_exp, rhs_join_exp);
	}

	// add the detached streams to the join op
	ExecutionPlan_AddOp(value_hash_join, left_branch);
	ExecutionPlan_AddOp(value_hash_join, right_branch);

	return value_hash_join;
}

// reduces a cartisian product to hash joins operations
static void _reduce_cp_to_hashjoin
(
	ExecutionPlan *plan,
	OpBase *cp
) {
	// retrieve all equality filter operations located upstream
	// from the Cartesian Product
	OpFilter **filter_ops = _locate_filters(cp);
	uint filter_count = array_len(filter_ops);

	// for each stream joined by the Cartesian product
	// collect all entities the stream resolves
	int stream_count = cp->childCount;
	rax *stream_entities[stream_count];

	//--------------------------------------------------------------------------
	// collect resolved variables
	//--------------------------------------------------------------------------

	for(uint i = 0; i < stream_count; i++) {
		stream_entities[i] = raxNew();
		ExecutionPlan_BoundVariables(cp->children[i], stream_entities[i],
				cp->children[i]->plan);
	}

	for(uint i = 0; i < filter_count; i++) {
		// try reduce the cartesian product to value hash join
		// with the current filter
		OpFilter *filter_op = filter_ops[i];

		// each filter being considered here tests for equality between
		// its left and right values
		// the Cartesian Product can be replaced if both sides of the filter
		// can be fully and separately resolved by exactly two child streams
		FT_FilterNode *f = filter_op->filterTree;

		// make sure LHS of the filter is resolved by a stream
		AR_ExpNode *lhs = f->pred.lhs;
		uint lhs_resolving_stream = _relate_exp_to_stream(lhs, stream_entities,
				stream_count);
		if(lhs_resolving_stream == NOT_RESOLVED) continue;

		// make sure RHS of the filter is resolved by a stream
		AR_ExpNode *rhs = f->pred.rhs;
		uint rhs_resolving_stream = _relate_exp_to_stream(rhs, stream_entities,
				stream_count);
		if(rhs_resolving_stream == NOT_RESOLVED) continue;

		// this filter is solved by a single cartesian product child
		// and needs to be propagated up
		if(lhs_resolving_stream == rhs_resolving_stream) {
			ExecutionPlan_RemoveOp(plan, (OpBase *)filter_op);
			ExecutionPlan_PushBelow(cp->children[rhs_resolving_stream], (OpBase *)filter_op);
			continue;
		}

		// clone the filter expressions
		lhs = AR_EXP_Clone(lhs);
		rhs = AR_EXP_Clone(rhs);

		// retrieve the relevant branch roots
		OpBase *right_branch = cp->children[rhs_resolving_stream];
		OpBase *left_branch = cp->children[lhs_resolving_stream];

		// detach the streams for the Value Hash Join from the execution plan
		ExecutionPlan_DetachOp(right_branch);
		ExecutionPlan_DetachOp(left_branch);

		// build hash join op
		OpBase *value_hash_join =
			_build_hash_join_op (cp->plan, left_branch, right_branch, lhs, rhs);

		// the filter will now be resolved by the join operation; remove it
		ExecutionPlan_RemoveOp(plan, (OpBase *)filter_op);
		OpBase_Free((OpBase *)filter_op);

		// place hash join op
		if(cp->childCount == 0) {
			// the entire Cartesian Product can be replaced with the join op
			ExecutionPlan_ReplaceOp(plan, cp, value_hash_join);
			OpBase_Free(cp);
			// the optimization has depleted all of the
			// cartesian product children, merged them and replaced the
			// cartesian product with the new operation
			// since the original cartesian product is no longer
			// a valid operation, and there might be additional filters
			// which are applicable to re position after the optimization is done
			// the following code tries to propagate up the remaining filters
			// and finish the loop
			break;
		} else {
			// the Cartesian Product still has a child operation
			// introduce the join op as another child
			ExecutionPlan_AddOp(cp, value_hash_join);
			// if there are remaining filters
			// re-collect cartesian product streams
			if(i + 1 < filter_count) {
				// streams are no longer valid since cartesian product changed
				for(int j = 0; j < stream_count; j++) {
					raxFree(stream_entities[j]);
				}

				stream_count = cp->childCount;

				for(int j = 0; j < stream_count; j++) {
					stream_entities[j] = raxNew();
					ExecutionPlan_BoundVariables(cp->children[j],
							stream_entities[j], cp->children[j]->plan);
				}
			}
		}
	}

	// clean up
	for(int i = 0; i < stream_count; i++) {
		raxFree(stream_entities[i]);
	}
	array_free(filter_ops);
}

// TODO: consider changing Cartesian Products such that each has exactly two
// child operations

// try to replace Cartesian Products (cross joins) with Value Hash Joins
// this is viable when a Cartesian Product is combining two streams that each
// satisfies one side of an EQUALS filter operation like:
// MATCH (a), (b) WHERE ID(a) = ID(b)
void applyJoin
(
	ExecutionPlan *plan
) {
	OpBase **cps = ExecutionPlan_CollectOps(plan->root,
			OPType_CARTESIAN_PRODUCT);
	uint cp_count = array_len(cps);

	for(uint i = 0; i < cp_count; i++) {
		OpBase *cp = cps[i];
		_reduce_cp_to_hashjoin(plan, cp);
	}

	array_free(cps);
}

