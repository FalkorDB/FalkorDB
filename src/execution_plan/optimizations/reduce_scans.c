/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../ops/op_filter.h"
#include "../ops/op_node_by_label_scan.h"
#include "../ops/op_conditional_traverse.h"
#include "../execution_plan_build/execution_plan_util.h"
#include "../execution_plan_build/execution_plan_modify.h"

// the reduce scans optimizer searches the execution plans for
// SCAN operations which set node N, in-case there's an earlier
// operation within the execution plan e.g. PROCEDURE-CALL which sets N
// then omit SCAN

static OpBase *_LabelScanToConditionalTraverse
(
	NodeByLabelScan *op
) {
	Graph *g = QueryCtx_GetGraph();
	const char *alias = op->n->alias;

	AlgebraicExpression *ae = AlgebraicExpression_NewOperand(GrB_NULL, true,
			alias, alias, NULL, op->n->label);

	return NewCondTraverseOp(op->op.plan, g, ae);
}

static void _reduceScans
(
	ExecutionPlan *plan,
	OpBase *scan
) {
	// return early if the scan has no child operations
	if(scan->childCount == 0) return;

	// the scan operation should be operating on a single alias
	ASSERT(array_len(scan->modifies) == 1);
	const char *alias = scan->modifies[0];

	// collect variables bound before this operation
	rax *bounded = raxNew();
	for(int i = 0; i < scan->childCount; i++) {
		ExecutionPlan_BoundVariables(scan->children[i], bounded,
				scan->children[i]->plan);
	}

	if(raxFind(bounded, (unsigned char *)alias, strlen(alias)) != raxNotFound) {
		// if the scanned alias is already bounded
		// then the scan operation is redundant
		if(scan->type == OPType_NODE_BY_LABEL_SCAN) {
			// if we are performing a label scan
			// introduce a conditional traversal to filter by label
			OpBase *traverse =
				_LabelScanToConditionalTraverse((NodeByLabelScan *)scan);
			ExecutionPlan_ReplaceOp(plan, scan, traverse);
		} else {
			// remove the redundant scan op
			ExecutionPlan_RemoveOp(plan, scan);
		}

		OpBase_Free(scan);
	}

	raxFree(bounded);
}

void reduceScans
(
	ExecutionPlan *plan
) {
	OPType op_types[] = {OPType_ALL_NODE_SCAN, OPType_NODE_BY_LABEL_SCAN};

	// collect all SCAN operations within the execution plan
	OpBase **scans = ExecutionPlan_CollectOpsMatchingTypes(plan->root, op_types,
			2);

	uint scan_count = array_len(scans);
	for(uint i = 0; i < scan_count; i++) {
		_reduceScans(plan, scans[i]);
	}

	array_free(scans);
}

