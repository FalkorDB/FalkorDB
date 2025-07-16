/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
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

/* The seek by ID optimization searches for a SCAN operation on which
 * a filter of the form ID(n) = X is applied in which case
 * both the SCAN and FILTER operations can be reduced into a single
 * NODE_BY_ID_SEEK operation. */

static bool _idFilter(FT_FilterNode *f, AST_Operator *rel, EntityID *id, bool *reverse) {
	if(f->t != FT_N_PRED) return false;
	if(f->pred.op == OP_NEQUAL) return false;

	AR_OpNode *op;
	AR_ExpNode *expr;
	AR_ExpNode *lhs = f->pred.lhs;
	AR_ExpNode *rhs = f->pred.rhs;
	*rel = f->pred.op;

	/* Either ID(N) compare const
	 * OR
	 * const compare ID(N) */
	if(lhs->type == AR_EXP_OPERAND && rhs->type == AR_EXP_OP) {
		op = &rhs->op;
		expr = lhs;
		*reverse = true;
	} else if(lhs->type == AR_EXP_OP && rhs->type == AR_EXP_OPERAND) {
		op = &lhs->op;
		expr = rhs;
		*reverse = false;
	} else {
		return false;
	}

	// Make sure ID is compared to a constant int64.
	SIValue val;
	bool reduced = AR_EXP_ReduceToScalar(expr, true, &val);
	if(!reduced) return false;
	if(SI_TYPE(val) != T_INT64) return false;
	*id = SI_GET_NUMERIC(val);

	// Make sure applied function is ID.
	if(strcasecmp(op->f->name, "id")) return false;

	return true;
}

static void _UseIdOptimization(ExecutionPlan *plan, OpBase *scan_op) {
	/* See if there's a filter of the form
	 * ID(n) op X
	 * where X is a constant and op in [EQ, GE, LE, GT, LT] */
	OpBase *parent = scan_op->parent;
	OpBase *grandparent;
	UnsignedRange *id_range = NULL;
	while(parent && parent->type == OPType_FILTER) {
		grandparent = parent->parent; // Track the next op to visit in case we free parent.
		OpFilter *filter = (OpFilter *)parent;
		FT_FilterNode *f = filter->filterTree;

		AST_Operator op;
		EntityID id;
		bool reverse;
		if(_idFilter(f, &op, &id, &reverse)) {
			if(!id_range) id_range = UnsignedRange_New();
			if(reverse) op = ArithmeticOp_ReverseOp(op);
			UnsignedRange_TightenRange(id_range, op, id);

			// Free replaced operations.
			ExecutionPlan_RemoveOp(plan, (OpBase *)filter);
			OpBase_Free((OpBase *)filter);
		}
		// Advance.
		parent = grandparent;
	}
	if(id_range) {
		/* Don't replace label scan, but set it to have range query.
		 * Issue 818 https://github.com/RedisGraph/RedisGraph/issues/818
		 * This optimization caused a range query over the entire range of ids in the graph
		 * regardless to the label. */
		if(scan_op->type == OPType_NODE_BY_LABEL_SCAN) {
			NodeByLabelScan *label_scan = (NodeByLabelScan *) scan_op;
			NodeByLabelScanOp_SetIDRange(label_scan, id_range);
		} else {
			const char *alias = ((AllNodeScan *)scan_op)->alias;
			OpBase *opNodeByIdSeek = NewNodeByIdSeekOp(scan_op->plan, alias, id_range);

			// Managed to reduce!
			ExecutionPlan_ReplaceOp(plan, scan_op, opNodeByIdSeek);
			OpBase_Free(scan_op);
		}
		UnsignedRange_Free(id_range);
	}
}

void seekByID(ExecutionPlan *plan) {
	ASSERT(plan != NULL);

	const OPType types[] = {OPType_ALL_NODE_SCAN, OPType_NODE_BY_LABEL_SCAN};
	OpBase **scan_ops = ExecutionPlan_CollectOpsMatchingTypes(plan->root, types, 2);

	for(int i = 0; i < array_len(scan_ops); i++) {
		_UseIdOptimization(plan, scan_ops[i]);
	}

	array_free(scan_ops);
}

