/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "../ops/ops.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../../arithmetic/aggregate_funcs/agg_funcs.h"
#include "../execution_plan_build/execution_plan_modify.h"

/* The reduceCount optimization will look for execution plan
 * performing solely node/edge counting: total number of nodes/edges
 * in the graph, total number of nodes/edges with a specific label/relation.
 * In which case we can avoid performing both SCAN* and AGGREGATE
 * operations by simply returning a precomputed count */

static int _identifyResultAndAggregateOps(OpBase *root, OpResult **opResult,
										  OpAggregate **opAggregate) {
	OpBase *op = root;
	// Op Results.
	if(op->type != OPType_RESULTS || op->childCount != 1) return 0;

	*opResult = (OpResult *)op;
	op = op->children[0];

	// Op Aggregate.
	if(op->type != OPType_AGGREGATE || op->childCount != 1) return 0;

	// Expecting a single aggregation, without ordering.
	*opAggregate = (OpAggregate *)op;
	if((*opAggregate)->aggregate_count != 1 || (*opAggregate)->key_count != 0) return 0;

	AR_ExpNode *exp = (*opAggregate)->aggregate_exps[0];

	// Make sure aggregation performs counting.
	if(exp->type != AR_EXP_OP ||
	   exp->op.f->aggregate != true ||
	   strcasecmp(AR_EXP_GetFuncName(exp), "count") ||
	   AR_EXP_PerformsDistinct(exp)) return 0;

	// Make sure Count acts on an alias.
	if(exp->op.child_count != 1) return 0;

	AR_ExpNode *arg = exp->op.children[0];
	return (arg->type == AR_EXP_OPERAND &&
			arg->operand.type == AR_EXP_VARIADIC);
}

/* Checks if execution plan solely performs node count */
static int _identifyNodeCountPattern(OpBase *root, OpResult **opResult, OpAggregate **opAggregate,
									 OpBase **opScan, const char **label) {
	// Reset.
	*label = NULL;
	*opScan = NULL;
	*opResult = NULL;
	*opAggregate = NULL;

	if(!_identifyResultAndAggregateOps(root, opResult, opAggregate)) return 0;
	OpBase *op = ((OpBase *)*opAggregate)->children[0];

	// Scan, either a full node or label scan.
	if((op->type != OPType_ALL_NODE_SCAN &&
		op->type != OPType_NODE_BY_LABEL_SCAN) ||
	   op->childCount != 0) {
		return 0;
	}

	*opScan = op;
	if(op->type == OPType_NODE_BY_LABEL_SCAN) {
		NodeByLabelScan *labelScan = (NodeByLabelScan *)op;
		*label = labelScan->n->label;
	}

	return 1;
}

bool _reduceNodeCount
(
	ExecutionPlan *plan
) {
	// we'll only modify execution plan if it is structured as follows:
	// "Scan -> Aggregate -> Results"
	const char *label;
	OpBase *opScan;
	OpResult *opResult;
	OpAggregate *opAggregate;

	// see if execution-plan matches the pattern:
	// "Scan -> Aggregate -> Results"
	// if that's not the case, simply return without making any modifications
	if(!_identifyNodeCountPattern(plan->root, &opResult, &opAggregate, &opScan,
				&label)) {
		return false;
	}

	// user is trying to get total number of nodes in the graph
	// optimize by skiping SCAN and AGGREGATE
	SIValue nodeCount;
	GraphContext *gc = QueryCtx_GetGraphCtx();

	// if label is specified, count only labeled entities
	if(label) {
		Schema *s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);
		if(s) nodeCount = SI_LongVal(Graph_LabeledNodeCount(gc->g, s->id));
		else nodeCount = SI_LongVal(0); // specified Label doesn't exists
	} else {
		nodeCount = SI_LongVal(Graph_NodeCount(gc->g));
	}

	// construct a constant expression, used by a new projection operation
	AR_ExpNode *exp = AR_EXP_NewConstOperandNode(nodeCount);
	// the new expression must be aliased to populate the Record
	exp->resolved_name = opAggregate->aggregate_exps[0]->resolved_name;
	AR_ExpNode **exps = array_new(AR_ExpNode *, 1);
	array_append(exps, exp);

	OpBase *opProject = NewProjectOp(opAggregate->op.plan, exps);

	// new execution plan: "Project -> Results"
	ExecutionPlan_RemoveOp(plan, opScan);
	OpBase_Free(opScan);

	ExecutionPlan_RemoveOp(plan, (OpBase *)opAggregate);
	OpBase_Free((OpBase *)opAggregate);

	ExecutionPlan_AddOp((OpBase *)opResult, opProject);
	return true;
}

// checks if execution plan solely performs edge count
static bool _identifyEdgeCountPattern
(
	OpBase *root,
	OpResult **opResult,
	OpAggregate **opAggregate,
	OpBase **opTraverse,
	OpBase **opScan
) {
	// reset
	*opScan      = NULL;
	*opTraverse  = NULL;
	*opResult    = NULL;
	*opAggregate = NULL;

	if(!_identifyResultAndAggregateOps(root, opResult, opAggregate)) {
		return false;
	}

	OpBase *op = ((OpBase *)*opAggregate)->children[0];

	if(op->type != OPType_CONDITIONAL_TRAVERSE || op->childCount != 1) {
		return false;
	}

	*opTraverse = op;
	op = op->children[0];

	// only a full node scan can be converted, as a labeled source acts as a
	// filter that may invalidate some of the edges
	if(op->type != OPType_ALL_NODE_SCAN || op->childCount != 0) return false;
	*opScan = op;

	return true;
}

void _reduceEdgeCount
(
	ExecutionPlan *plan
) {
	// we'll only modify execution plan if it is structured as follows:
	// "Full Scan -> Conditional Traverse -> Aggregate -> Results"
	OpBase *opScan;
	OpBase *opTraverse;
	OpResult *opResult;
	OpAggregate *opAggregate;

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
	OpCondTraverse *condTraverse = (OpCondTraverse *)opTraverse;
	// the traversal op doesn't contain information about the traversed edge,
	// cannot apply optimization
	if(!condTraverse->edge_ctx) return;

	uint relationCount = array_len(condTraverse->edge_ctx->edgeRelationTypes);

	uint64_t edges = 0;
	for(uint i = 0; i < relationCount; i++) {
		int relType = condTraverse->edge_ctx->edgeRelationTypes[i];
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

	OpBase *opProject = NewProjectOp(opAggregate->op.plan, exps);

	// new execution plan: "Project -> Results"
	ExecutionPlan_RemoveOp(plan, opScan);
	OpBase_Free(opScan);

	ExecutionPlan_RemoveOp(plan, (OpBase *)opTraverse);
	OpBase_Free(opTraverse);

	ExecutionPlan_RemoveOp(plan, (OpBase *)opAggregate);
	OpBase_Free((OpBase *)opAggregate);

	ExecutionPlan_AddOp((OpBase *)opResult, opProject);
}

// checks if execution plan solely performs cartesian product count
static bool _identifyCartesianProductCountPattern
(
	OpBase *root,
	OpResult **opResult,
	OpAggregate **opAggregate,
	OpBase **opCartesian
) {
	// reset
	*opResult = NULL;
	*opAggregate = NULL;
	*opCartesian = NULL;

	if(!_identifyResultAndAggregateOps(root, opResult, opAggregate)) {
		return false;
	}

	OpBase *op = ((OpBase *)*opAggregate)->children[0];

	if(op->type != OPType_CARTESIAN_PRODUCT || op->childCount < 2) {
		return false;
	}

	*opCartesian = op;

	// check that all children of the cartesian product are simple scans
	// or simple aggregations over scans (for the nested case)
	for(int i = 0; i < op->childCount; i++) {
		OpBase *child = op->children[i];
		
		if(child->type == OPType_ALL_NODE_SCAN || 
		   child->type == OPType_NODE_BY_LABEL_SCAN) {
			// Simple scan case - just make sure it has no children
			if(child->childCount != 0) {
				return false;
			}
		} else if(child->type == OPType_AGGREGATE) {
			// Nested aggregation case - check it's a simple count over cartesian product
			OpAggregate *nestedAgg = (OpAggregate *)child;
			if(nestedAgg->aggregate_count != 1 || nestedAgg->key_count != 0) {
				return false;
			}

			AR_ExpNode *exp = nestedAgg->aggregate_exps[0];
			// Make sure aggregation performs counting
			if(exp->type != AR_EXP_OP ||
			   exp->op.f->aggregate != true ||
			   strcasecmp(AR_EXP_GetFuncName(exp), "count") ||
			   AR_EXP_PerformsDistinct(exp)) {
				return false;
			}

			// Check that the nested aggregate has a cartesian product child
			if(child->childCount != 1) return false;
			OpBase *nestedCartesian = child->children[0];
			if(nestedCartesian->type != OPType_CARTESIAN_PRODUCT) return false;

			// Check that the nested cartesian product only has scan children
			for(int j = 0; j < nestedCartesian->childCount; j++) {
				OpBase *scanChild = nestedCartesian->children[j];
				if(scanChild->type != OPType_ALL_NODE_SCAN && 
				   scanChild->type != OPType_NODE_BY_LABEL_SCAN) {
					return false;
				}
				if(scanChild->childCount != 0) {
					return false;
				}
			}
		} else {
			// Unsupported child type
			return false;
		}
	}

	return true;
}

// Helper function to calculate node count for a scan operation
static uint64_t _getNodeCountForScan(OpBase *scanOp, GraphContext *gc) {
	if(scanOp->type == OPType_ALL_NODE_SCAN) {
		return Graph_NodeCount(gc->g);
	} else if(scanOp->type == OPType_NODE_BY_LABEL_SCAN) {
		NodeByLabelScan *labelScan = (NodeByLabelScan *)scanOp;
		const char *label = labelScan->n->label;
		if(label) {
			Schema *s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);
			if(s) {
				return Graph_LabeledNodeCount(gc->g, s->id);
			} else {
				return 0; // specified Label doesn't exist
			}
		} else {
			return Graph_NodeCount(gc->g);
		}
	}
	return 0;
}

// Helper function to calculate cartesian product count for all children
static uint64_t _calculateCartesianProductCount(OpBase *cartesianOp, GraphContext *gc) {
	uint64_t totalCount = 1;

	for(int i = 0; i < cartesianOp->childCount; i++) {
		OpBase *child = cartesianOp->children[i];
		
		if(child->type == OPType_ALL_NODE_SCAN || 
		   child->type == OPType_NODE_BY_LABEL_SCAN) {
			// Simple scan case
			uint64_t nodeCount = _getNodeCountForScan(child, gc);
			totalCount *= nodeCount;
		} else if(child->type == OPType_AGGREGATE) {
			// Nested aggregation case - calculate the count of its cartesian product
			OpBase *nestedCartesian = child->children[0];
			uint64_t nestedCount = _calculateCartesianProductCount(nestedCartesian, gc);
			totalCount *= nestedCount;
		}
	}

	return totalCount;
}

bool _reduceCartesianProductCount
(
	ExecutionPlan *plan
) {
	// we'll only modify execution plan if it is structured as follows:
	// "Node Scans -> Cartesian Product -> Aggregate -> Results"
	// or nested version with embedded aggregations
	OpBase *opCartesian;
	OpResult *opResult;
	OpAggregate *opAggregate;

	// see if execution-plan matches the pattern
	if(!_identifyCartesianProductCountPattern(plan->root, &opResult, &opAggregate, &opCartesian)) {
		return false;
	}

	// calculate the cartesian product count
	GraphContext *gc = QueryCtx_GetGraphCtx();
	uint64_t totalCount = _calculateCartesianProductCount(opCartesian, gc);

	SIValue cartesianCount = SI_LongVal(totalCount);

	// construct a constant expression, used by a new projection operation
	AR_ExpNode *exp = AR_EXP_NewConstOperandNode(cartesianCount);
	// the new expression must be aliased to populate the Record
	exp->resolved_name = opAggregate->aggregate_exps[0]->resolved_name;
	AR_ExpNode **exps = array_new(AR_ExpNode *, 1);
	array_append(exps, exp);

	OpBase *opProject = NewProjectOp(opAggregate->op.plan, exps);

	// Remove the cartesian product and its children operations
	// First, collect all operations that need to be removed
	OpBase **toRemove = array_new(OpBase *, 16);
	
	// Add the main cartesian product
	array_append(toRemove, opCartesian);
	
	// Add all its children (scans and nested aggregations)
	for(int i = 0; i < opCartesian->childCount; i++) {
		OpBase *child = opCartesian->children[i];
		array_append(toRemove, child);
		
		// If child is an aggregate with a nested cartesian product, add those too
		if(child->type == OPType_AGGREGATE && child->childCount == 1) {
			OpBase *nestedCartesian = child->children[0];
			if(nestedCartesian->type == OPType_CARTESIAN_PRODUCT) {
				array_append(toRemove, nestedCartesian);
				// Add all nested scan operations
				for(int j = 0; j < nestedCartesian->childCount; j++) {
					array_append(toRemove, nestedCartesian->children[j]);
				}
			}
		}
	}
	
	// Remove all collected operations
	for(uint i = 0; i < array_len(toRemove); i++) {
		ExecutionPlan_RemoveOp(plan, toRemove[i]);
		OpBase_Free(toRemove[i]);
	}
	array_free(toRemove);
	
	// Remove the main aggregate operation
	ExecutionPlan_RemoveOp(plan, (OpBase *)opAggregate);
	OpBase_Free((OpBase *)opAggregate);

	// Add the project operation
	ExecutionPlan_AddOp((OpBase *)opResult, opProject);
	return true;
}

void reduceCount
(
	ExecutionPlan *plan
) {
	// start by trying to identify node count pattern
	// if unsuccessful try edge count pattern
	// if unsuccessful try cartesian product count pattern
	if(!_reduceNodeCount(plan)) {
		if(!_reduceEdgeCount(plan)) {
			_reduceCartesianProductCount(plan);
		}
	}
}

