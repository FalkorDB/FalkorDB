/*
* Copyright 2018-2021 Redis Labs Ltd. and Contributors
*
* This file is available under the Redis Labs Source Available License Agreement
*/

#include "../runtimes/interpreted/ops/ops.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../../arithmetic/aggregate_funcs/agg_funcs.h"
#include "../runtimes/interpreted/execution_plan_build/runtime_execution_plan_modify.h"

/* The reduceCount optimization will look for execution plan
 * performing solely node/edge counting: total number of nodes/edges
 * in the graph, total number of nodes/edges with a specific label/relation.
 * In which case we can avoid performing both SCAN* and AGGREGATE
 * operations by simply returning a precomputed count */

static int _identifyResultAndAggregateOps(RT_OpBase *root, RT_OpResult **opResult,
										  RT_OpAggregate **opAggregate) {
	RT_OpBase *op = root;
	// Op Results.
	if(op->op_desc->type != OPType_RESULTS || op->childCount != 1) return 0;

	*opResult = (RT_OpResult *)op;
	op = op->children[0];

	// Op Aggregate.
	if(op->op_desc->type != OPType_AGGREGATE || op->childCount != 1) return 0;

	// Expecting a single aggregation, without ordering.
	*opAggregate = (RT_OpAggregate *)op;
	if((*opAggregate)->op_desc->aggregate_count != 1 || (*opAggregate)->op_desc->key_count != 0) return 0;

	AR_ExpNode *exp = (*opAggregate)->aggregate_exps[0];

	// Make sure aggregation performs counting.
	if(exp->type != AR_EXP_OP ||
	   exp->op.f->aggregate != true ||
	   strcasecmp(exp->op.func_name, "count") ||
	   Aggregate_PerformsDistinct(exp->op.f->privdata)) return 0;

	// Make sure Count acts on an alias.
	if(exp->op.child_count != 1) return 0;

	AR_ExpNode *arg = exp->op.children[0];
	return (arg->type == AR_EXP_OPERAND &&
			arg->operand.type == AR_EXP_VARIADIC);
}

/* Checks if execution plan solely performs node count */
static int _identifyNodeCountPattern(RT_OpBase *root, RT_OpResult **opResult, RT_OpAggregate **opAggregate,
									 RT_OpBase **opScan, const char **label) {
	// Reset.
	*label = NULL;
	*opScan = NULL;
	*opResult = NULL;
	*opAggregate = NULL;

	if(!_identifyResultAndAggregateOps(root, opResult, opAggregate)) return 0;
	RT_OpBase *op = ((RT_OpBase *)*opAggregate)->children[0];

	// Scan, either a full node or label scan.
	if((op->op_desc->type != OPType_ALL_NODE_SCAN &&
		op->op_desc->type != OPType_NODE_BY_LABEL_SCAN) ||
	    op->childCount != 0) {
		return 0;
	}

	*opScan = op;
	if(op->op_desc->type == OPType_NODE_BY_LABEL_SCAN) {
		RT_NodeByLabelScan *labelScan = (RT_NodeByLabelScan *)op;
		*label = labelScan->op_desc->n.label;
	}

	return 1;
}

bool _reduceNodeCount(RT_ExecutionPlan *plan) {
	/* We'll only modify execution plan if it is structured as follows:
	 * "Scan -> Aggregate -> Results" */
	const char *label;
	RT_OpBase *opScan;
	RT_OpResult *opResult;
	RT_OpAggregate *opAggregate;

	/* See if execution-plan matches the pattern:
	 * "Scan -> Aggregate -> Results".
	 * if that's not the case, simply return without making any modifications. */
	if(!_identifyNodeCountPattern(plan->root, &opResult, &opAggregate, &opScan, &label)) return false;

	/* User is trying to get total number of nodes in the graph
	 * optimize by skiping SCAN and AGGREGATE. */
	SIValue nodeCount;
	GraphContext *gc = QueryCtx_GetGraphCtx();

	// If label is specified, count only labeled entities.
	if(label) {
		Schema *s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);
		if(s) nodeCount = SI_LongVal(Graph_LabeledNodeCount(gc->g, s->id));
		else nodeCount = SI_LongVal(0); // Specified Label doesn't exists.
	} else {
		nodeCount = SI_LongVal(Graph_NodeCount(gc->g));
	}

	// Construct a constant expression, used by a new projection operation
	AR_ExpNode *exp = AR_EXP_NewConstOperandNode(nodeCount);
	// The new expression must be aliased to populate the Record.
	exp->resolved_name = opAggregate->aggregate_exps[0]->resolved_name;
	AR_ExpNode **exps = array_new(AR_ExpNode *, 1);
	array_append(exps, exp);

	OpProject *opProject = (OpProject *)NewProjectOp(opAggregate->op_desc->op.plan, exps);
	RT_OpBase *rt_opProject = RT_NewProjectOp(opAggregate->op.plan, opProject);

	// New execution plan: "Project -> Results"
	RT_ExecutionPlan_RemoveOp(plan, opScan);
	RT_OpBase_Free(opScan);

	RT_ExecutionPlan_RemoveOp(plan, (RT_OpBase *)opAggregate);
	RT_OpBase_Free((RT_OpBase *)opAggregate);

	RT_ExecutionPlan_AddOp((RT_OpBase *)opResult, rt_opProject);
	return true;
}

/* Checks if execution plan solely performs edge count */
static bool _identifyEdgeCountPattern(RT_OpBase *root, RT_OpResult **opResult,
		RT_OpAggregate **opAggregate, RT_OpBase **opTraverse, RT_OpBase **opScan) {

	// reset
	*opScan = NULL;
	*opTraverse = NULL;
	*opResult = NULL;
	*opAggregate = NULL;

	if(!_identifyResultAndAggregateOps(root, opResult, opAggregate)) {
		return false;
	}

	RT_OpBase *op = ((RT_OpBase *)*opAggregate)->children[0];

	if(op->op_desc->type != OPType_CONDITIONAL_TRAVERSE || op->childCount != 1) {
		return false;
	}

	*opTraverse = op;
	op = op->children[0];

	// only a full node scan can be converted, as a labeled source acts as a
	// filter that may invalidate some of the edges
	if(op->op_desc->type != OPType_ALL_NODE_SCAN || op->childCount != 0) return false;
	*opScan = op;

	return true;
}

void _reduceEdgeCount(RT_ExecutionPlan *plan) {
	// we'll only modify execution plan if it is structured as follows:
	// "Full Scan -> Conditional Traverse -> Aggregate -> Results"
	RT_OpBase *opScan;
	RT_OpBase *opTraverse;
	RT_OpResult *opResult;
	RT_OpAggregate *opAggregate;

	// see if execution-plan matches the pattern:
	// "Full Scan -> Conditional Traverse -> Aggregate -> Results"
	// if that's not the case, simply return without making any modifications
	if(!_identifyEdgeCountPattern(plan->root, &opResult, &opAggregate,
				&opTraverse, &opScan)) return;

	// user is trying to count edges (either in total or of specific types)
	// in the graph. optimize by skipping Scan, Traverse and Aggregate
	Graph *g = QueryCtx_GetGraph();
	SIValue edgeCount = SI_LongVal(0);

	// if type is specified, count only labeled entities
	RT_OpCondTraverse *condTraverse = (RT_OpCondTraverse *)opTraverse;

	const char *edge = AlgebraicExpression_Edge(condTraverse->ae);
	// the traversal op doesn't contain information about the traversed edge,
	// cannot apply optimization
	if(!edge) return;

	QGEdge *e = QueryGraph_GetEdgeByAlias(plan->plan_desc->query_graph, edge);
	
	uint relationCount = array_len(e->reltypeIDs);

	uint64_t edges = 0;
	if(relationCount == 0) {
		// should be the only relationship type mentioned, -[]->
		edges = Graph_EdgeCount(g);
	}
	for(uint i = 0; i < relationCount; i++) {
		int relType = e->reltypeIDs[i];
		switch(relType) {
			case GRAPH_NO_RELATION:
				// should be the only relationship type mentioned, -[]->
				edges = Graph_EdgeCount(g);
				break;
			case GRAPH_UNKNOWN_RELATION:
				// no change to current count, -[:none_existing]->
				break;
			default:
				edges += Graph_RelationEdgeCount(g, relType);
		}
	}
	edgeCount = SI_LongVal(edges);

	// construct a constant expression, used by a new projection operation
	AR_ExpNode *exp = AR_EXP_NewConstOperandNode(edgeCount);
	// the new expression must be aliased to populate the Record
	exp->resolved_name = opAggregate->aggregate_exps[0]->resolved_name;
	AR_ExpNode **exps = array_new(AR_ExpNode *, 1);
	array_append(exps, exp);

	OpProject *opProject = (OpProject *)NewProjectOp(opAggregate->op_desc->op.plan, exps);
	RT_OpBase *rt_opProject = RT_NewProjectOp(opAggregate->op.plan, opProject);

	// new execution plan: "Project -> Results"
	RT_ExecutionPlan_RemoveOp(plan, opScan);
	RT_OpBase_Free(opScan);

	RT_ExecutionPlan_RemoveOp(plan, (RT_OpBase *)opTraverse);
	RT_OpBase_Free(opTraverse);

	RT_ExecutionPlan_RemoveOp(plan, (RT_OpBase *)opAggregate);
	RT_OpBase_Free((RT_OpBase *)opAggregate);

	RT_ExecutionPlan_AddOp((RT_OpBase *)opResult, rt_opProject);
}

void reduceCount(RT_ExecutionPlan *plan) {
	// start by trying to identify node count pattern
	// if unsuccessful try edge count pattern
	if(!_reduceNodeCount(plan)) _reduceEdgeCount(plan);
}
