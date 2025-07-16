/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "op_edge_by_index_scan.h"
#include "../../query_ctx.h"
#include "shared/print_functions.h"
#include "../../filter_tree/ft_to_rsq.h"

// forward declarations
static OpResult EdgeIndexScanInit(OpBase *opBase);
static Record EdgeIndexScanConsume(OpBase *opBase);
static Record EdgeIndexScanConsumeFromChild(OpBase *opBase);
static OpResult EdgeIndexScanReset(OpBase *opBase);
static void EdgeIndexScanFree(OpBase *opBase);

static void EdgeIndexScanToString
(
	const OpBase *ctx,
	sds *buf
) {
	ASSERT(ctx != NULL);
	ASSERT(buf != NULL);

	OpEdgeIndexScan *op = (OpEdgeIndexScan *)ctx;

	// Edge By Index Scan | [e:R]
	*buf = sdscatprintf(*buf, "%s | ", ctx->name);
	QGEdge_ToString(op->edge, buf);
}

OpBase *NewEdgeIndexScanOp
(
	const ExecutionPlan *plan,
	Graph *g,
	QGEdge *e,
	RSIndex *idx,
	FT_FilterNode *filter
) {
	// validate inputs
	ASSERT(g                       != NULL);
	ASSERT(e                       != NULL);
	ASSERT(idx                     != NULL);
	ASSERT(plan                    != NULL);
	ASSERT(filter                  != NULL);
	ASSERT(QGEdge_Alias(e)         != NULL);
	ASSERT(QGEdge_RelationCount(e) == 1);

	OpEdgeIndexScan *op      =  rm_malloc(sizeof(OpEdgeIndexScan));
	op->g                    =  g;
	op->idx                  =  idx;
	op->edge                 =  e;
	op->iter                 =  NULL;
	op->filter               =  filter;
	op->child_record         =  NULL;
	op->current_src_node_id  =  NULL;
	op->current_dest_node_id =  NULL;
	op->unresolved_filters   =  NULL;
	op->rebuild_index_query  =  false;

	// set our Op operations
	OpBase_Init(
			(OpBase *)op,
			OPType_EDGE_BY_INDEX_SCAN,
			"Edge By Index Scan",
			EdgeIndexScanInit,
			EdgeIndexScanConsume,
			EdgeIndexScanReset,
			EdgeIndexScanToString,
			NULL,
			EdgeIndexScanFree,
			false,
			plan);

	op->edgeRecIdx = OpBase_Modifies((OpBase *)op, QGEdge_Alias(e));

	return (OpBase *)op;
}

static OpResult EdgeIndexScanInit
(
	OpBase *opBase
) {
	OpEdgeIndexScan *op = (OpEdgeIndexScan *)opBase;

	// source and destination nodes may or may not already be resolved
	// missing nodes will be resolved by this operation
	const  char  *src_alias   =  QGNode_Alias(QGEdge_Src(op->edge));
	const  char  *dest_alias  =  QGNode_Alias(QGEdge_Dest(op->edge));

	op->srcAware   =  OpBase_ChildrenAware((OpBase *)op, src_alias, &op->srcRecIdx);
	op->destAware  =  OpBase_ChildrenAware((OpBase *)op, dest_alias, &op->destRecIdx);

	if(!op->srcAware) {
		op->srcRecIdx = OpBase_Modifies((OpBase *)op, src_alias);
	}

	if(!op->destAware) {
		op->destRecIdx = OpBase_Modifies((OpBase *)op, dest_alias);
	}

	if(opBase->childCount > 0) {
		const char *alias =  QGEdge_Alias(op->edge);
		if(op->srcAware) {
			op->current_src_node_id  = AR_EXP_NewConstOperandNode(SI_NullVal());
			FT_FilterNode *ft = FilterTree_CreatePredicateFilter(OP_EQUAL, 
				AR_EXP_NewAttributeAccessNode(AR_EXP_NewVariableOperandNode(alias), "_src_id"), 
				op->current_src_node_id);
			FT_FilterNode *root = FilterTree_CreateConditionFilter(OP_AND);
			FilterTree_AppendLeftChild(root, op->filter);
			FilterTree_AppendRightChild(root, ft);
			op->filter = root;
			op->rebuild_index_query = true;
		}

		if(op->destAware) {
			op->current_dest_node_id  = AR_EXP_NewConstOperandNode(SI_NullVal());
			FT_FilterNode *ft = FilterTree_CreatePredicateFilter(OP_EQUAL, 
				AR_EXP_NewAttributeAccessNode(AR_EXP_NewVariableOperandNode(alias), "_dest_id"), 
				op->current_dest_node_id);
			FT_FilterNode *root = FilterTree_CreateConditionFilter(OP_AND);
			FilterTree_AppendLeftChild(root, op->filter);
			FilterTree_AppendRightChild(root, ft);
			op->filter = root;
			op->rebuild_index_query = true;
		}

		if(!op->rebuild_index_query) {
			// find out how many different entities are refered to 
			// within the filter tree, if number of entities equals 1
			// (current edge being scanned) there's no need to re-build the index
			// query for every input record
			rax *entities = FilterTree_CollectModified(op->filter);
			op->rebuild_index_query = raxSize(entities) > 1;
			raxFree(entities);
		}

		OpBase_UpdateConsume(opBase, EdgeIndexScanConsumeFromChild);
	}

	return OP_OK;
}

static inline void _UpdateRecord
(
	OpEdgeIndexScan *op,          // executing operation
	Record r,                     // record to update
	const EdgeIndexKey *edge_key  // value retrieved from the index
) {
	// populate Record with edge, src and destination node
	int res;
	UNUSED(res);
	
	Edge e = GE_NEW_LABELED_EDGE(op->edge->reltypes[0], op->edge->reltypeIDs[0]);

	e.src_id   =  edge_key->src_id;
	e.dest_id  =  edge_key->dest_id;
	EntityID  edge_id  =  edge_key->edge_id;

	res = Graph_GetEdge(op->g, edge_id, &e);
	ASSERT(res != 0);

	if(!op->srcAware) {
		Node src = GE_NEW_NODE();
		res = Graph_GetNode(op->g, e.src_id, &src);
		ASSERT(res != 0);
		Record_AddNode(r, op->srcRecIdx, src);
	}

	if(!op->destAware) {
		Node dest = GE_NEW_NODE();
		res = Graph_GetNode(op->g, e.dest_id, &dest);
		ASSERT(res != 0);
		Record_AddNode(r, op->destRecIdx, dest);
	}

	Record_AddEdge(r, op->edgeRecIdx, e);
}

static inline bool _PassUnresolvedFilters
(
	const OpEdgeIndexScan *op,
	Record r
) {
	FT_FilterNode *unresolved_filters = op->unresolved_filters;
	if(unresolved_filters == NULL) return true; // no filters

	return FilterTree_applyFilters(unresolved_filters, r) == FILTER_PASS;
}

static void UpdateCurrentAwareIds(const OpEdgeIndexScan *op) {
	if(op->current_src_node_id) {
		Node *n = Record_GetNode(op->child_record, op->srcRecIdx);
		op->current_src_node_id->operand.constant = SI_LongVal(ENTITY_GET_ID(n));
	}

	if(op->current_dest_node_id) {
		Node *n = Record_GetNode(op->child_record, op->destRecIdx);
		op->current_dest_node_id->operand.constant = SI_LongVal(ENTITY_GET_ID(n));
	}
}

static Record EdgeIndexScanConsumeFromChild
(
	OpBase *opBase
) {
	OpEdgeIndexScan	*op = (OpEdgeIndexScan*)opBase;
	const EdgeIndexKey *edgeKey = NULL;

pull_index:
	//--------------------------------------------------------------------------
	// pull from index
	//--------------------------------------------------------------------------

	if(op->iter != NULL && op->child_record != NULL) {
		while((edgeKey = RediSearch_ResultsIteratorNext(op->iter, op->idx, NULL))
				!= NULL) {
			// populate record with edge
			_UpdateRecord(op, op->child_record, edgeKey);
			// apply unresolved filters
			if(_PassUnresolvedFilters(op, op->child_record)) {
				// clone the held Record, as it will be freed upstream
				return OpBase_CloneRecord(op->child_record);
			}
		}
	}

	//--------------------------------------------------------------------------
	// index depleted
	//--------------------------------------------------------------------------

	// free input record
	if(op->child_record != NULL) {
		OpBase_DeleteRecord(op->child_record);
		op->child_record = NULL;
	}

	//--------------------------------------------------------------------------
	// pull from child
	//--------------------------------------------------------------------------

	op->child_record = OpBase_Consume(op->op.children[0]);
	if(op->child_record == NULL) return NULL; // depleted

	//--------------------------------------------------------------------------
	// reset index iterator
	//--------------------------------------------------------------------------

	if(op->rebuild_index_query) {
		// free previous iterator
		if(op->iter != NULL) {
			RediSearch_ResultsIteratorFree(op->iter);
			op->iter = NULL;
		}

		// free previous unresolved filters
		if(op->unresolved_filters != NULL) {
			FilterTree_Free(op->unresolved_filters);
			op->unresolved_filters = NULL;
		}

		UpdateCurrentAwareIds(op);

		// rebuild index query, probably relies on runtime values
		// resolve runtime variables within filter
		FT_FilterNode *filter = FilterTree_Clone(op->filter);
		FilterTree_ResolveVariables(filter, op->child_record);

		// make sure there's only one unresolve entity in filter
		#ifdef RG_DEBUG
		{
			rax *entities = FilterTree_CollectModified(filter);
			ASSERT(raxSize(entities) == 1);
			raxFree(entities);
		}
		#endif

		// convert filter into a RediSearch query
		RSQNode *rs_query_node = FilterTreeToQueryNode(&op->unresolved_filters,
				filter, op->idx);
		FilterTree_Free(filter);

		// create iterator
		ASSERT(rs_query_node != NULL);
		op->iter = RediSearch_GetResultsIterator(rs_query_node, op->idx);
	} else {
		// build index query only once (first call)
		// reset it if already initialized
		if(op->iter == NULL) {
			// first call to consume, create query and iterator
			RSQNode *rs_query_node = FilterTreeToQueryNode(
					&op->unresolved_filters, op->filter, op->idx);
			ASSERT(rs_query_node != NULL);
			op->iter = RediSearch_GetResultsIterator(rs_query_node, op->idx);
		} else {
			// reset existing iterator
			RediSearch_ResultsIteratorReset(op->iter);
		}
	}

	// repull from index
	goto pull_index;
}

static Record EdgeIndexScanConsume
(
	OpBase *opBase
) {
	OpEdgeIndexScan *op = (OpEdgeIndexScan *)opBase;

	// create iterator on first call
	if(op->iter == NULL) {
		UpdateCurrentAwareIds(op);

		RSQNode *rs_query_node = FilterTreeToQueryNode(&op->unresolved_filters,
				op->filter, op->idx);

		op->iter = RediSearch_GetResultsIterator(rs_query_node, op->idx);
	}

	const EdgeIndexKey *edgeKey = NULL;

	// populate the Record with the actual edge
	Record r = OpBase_CreateRecord((OpBase *)op);
	while((edgeKey = RediSearch_ResultsIteratorNext(op->iter, op->idx, NULL))
			!= NULL) {
		// populate record with edge
		_UpdateRecord(op, r, edgeKey);
		// apply unresolved filters
		if(_PassUnresolvedFilters(op, r)) {
			return r;
		}
	}

	OpBase_DeleteRecord(r);

	return NULL;
}

static OpResult EdgeIndexScanReset(OpBase *opBase) {
	OpEdgeIndexScan *op = (OpEdgeIndexScan *)opBase;

	if(op->iter) {
		RediSearch_ResultsIteratorFree(op->iter);
		op->iter = NULL;
	}

	if(op->unresolved_filters) {
		FilterTree_Free(op->unresolved_filters);
		op->unresolved_filters = NULL;
	}

	return OP_OK;
}

static void EdgeIndexScanFree(OpBase *opBase) {
	OpEdgeIndexScan *op = (OpEdgeIndexScan *)opBase;
	// as long as this Index iterator is alive the index is
	// read locked, if this index scan operation is part of
	// a query which will modified this index we'll be stuck in
	// a dead lock, as we're unable to acquire index write lock
	if(op->iter) {
		RediSearch_ResultsIteratorFree(op->iter);
		op->iter = NULL;
	}

	if(op->child_record) {
		OpBase_DeleteRecord(op->child_record);
		op->child_record = NULL;
	}

	if(op->filter) {
		FilterTree_Free(op->filter);
		op->filter = NULL;
	}

	if(op->unresolved_filters) {
		FilterTree_Free(op->unresolved_filters);
		op->unresolved_filters = NULL;
	}
}

