/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "../../util/arr.h"
#include "../ops/op_filter.h"
#include "../ops/op_all_node_scan.h"
#include "../ops/op_node_by_id_seek.h"
#include "../ops/op_node_by_label_scan.h"
#include "../../util/range/numeric_range.h"
#include "../../arithmetic/arithmetic_op.h"
#include "../execution_plan_build/execution_plan_util.h"
#include "../execution_plan_build/execution_plan_modify.h"

// the seek by ID optimization searches for a SCAN operation on which
// a filter of the form ID(n) = X is applied in which case
// both the SCAN and FILTER operations can be reduced into a single
// NODE_BY_ID_SEEK operation

static bool _idFilter
(
	FT_FilterNode *f,
	AST_Operator *rel,
	AR_ExpNode **id_exp,
	bool *reverse
) {
	if(f->t       != FT_N_PRED) return false;
	if(f->pred.op == OP_NEQUAL) return false;

	AR_OpNode *op;
	AR_ExpNode *lhs = f->pred.lhs;
	AR_ExpNode *rhs = f->pred.rhs;
	*rel = f->pred.op;

	// either ID(N) compare const
	// OR
	// const compare ID(N)
	if(lhs->type == AR_EXP_OPERAND && rhs->type == AR_EXP_OP) {
		op = &rhs->op;
		*id_exp = lhs;
		*reverse = true;
	} else if(lhs->type == AR_EXP_OP && rhs->type == AR_EXP_OPERAND) {
		op = &lhs->op;
		*id_exp = rhs;
		*reverse = false;
	} else {
		return false;
	}

	// make sure applied function is ID.
	if(strcasecmp(op->f->name, "id")) return false;

	return true;
}

static void _UseIdOptimization
(
	ExecutionPlan *plan,
	OpBase *scan_op
) {
	// see if there's a filter of the form
	// ID(n) op X
	// where X is a constant and op in [EQ, GE, LE, GT, LT]
	OpBase *grandparent;
	OpBase *parent = scan_op->parent;
	RangeExpression *ranges = array_new(RangeExpression, 1);
	while(parent && parent->type == OPType_FILTER) {
		// track the next op to visit in case we free parent
		grandparent = parent->parent;
		OpFilter *filter = (OpFilter *)parent;
		FT_FilterNode *f = filter->filterTree;

		bool         reverse;
		AR_ExpNode  *id_exp;
		AST_Operator op;

		if(_idFilter(f, &op, &id_exp, &reverse)) {
			if(reverse) op = ArithmeticOp_ReverseOp(op);
			id_exp = AR_EXP_Clone(id_exp);

			// Free replaced operations.
			ExecutionPlan_RemoveOp(plan, (OpBase *)filter);
			array_append(ranges, ((RangeExpression){.op = op, .exp = id_exp}));
			OpBase_Free((OpBase *)filter);
		}
		// advance
		parent = grandparent;
	}
	if(array_len(ranges) > 0) {
		/* Don't replace label scan, but set it to have range query.
		 * Issue 818 https://github.com/RedisGraph/RedisGraph/issues/818
		 * This optimization caused a range query over the entire range of ids in the graph
		 * regardless to the label. */
		if(scan_op->type == OPType_NODE_BY_LABEL_SCAN) {
			NodeByLabelScan *label_scan = (NodeByLabelScan *) scan_op;
			NodeByLabelScanOp_SetIDRange(label_scan, ranges);
		} else {
			const char *alias = ((AllNodeScan *)scan_op)->alias;
			OpBase *opNodeByIdSeek = NewNodeByIdSeekOp(scan_op->plan, alias, ranges);

			// Managed to reduce!
			ExecutionPlan_ReplaceOp(plan, scan_op, opNodeByIdSeek);
			OpBase_Free(scan_op);
		}
	} else {
		array_free(ranges);
	}
}

void seekByID
(
	ExecutionPlan *plan
) {
	ASSERT(plan != NULL);

	const OPType types[] = {OPType_ALL_NODE_SCAN, OPType_NODE_BY_LABEL_SCAN};
	OpBase **scan_ops = ExecutionPlan_CollectOpsMatchingTypes(plan->root, types, 2);

	for(int i = 0; i < array_len(scan_ops); i++) {
		_UseIdOptimization(plan, scan_ops[i]);
	}

	array_free(scan_ops);
}

