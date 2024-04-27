/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "op_node_by_index_scan.h"
#include "../../query_ctx.h"
#include "../../index/index.h"
#include "shared/print_functions.h"

// forward declarations
static OpResult IndexScanInit(OpBase *opBase);
static Record IndexScanConsume(OpBase *opBase);
static Record IndexScanConsumeFromChild(OpBase *opBase);
static OpResult IndexScanReset(OpBase *opBase);
static void IndexScanFree(OpBase *opBase);

static void IndexScanToString(const OpBase *ctx, sds *buf) {
	IndexScan *op = (IndexScan *)ctx;
	ScanToString(ctx, buf, op->n->alias, op->n->label);
}

OpBase *NewIndexScanOp
(
	const ExecutionPlan *plan,
	Graph *g,
	NodeScanCtx *n,
	Index idx,
	FT_FilterNode *filter
) {
	// validate inputs
	ASSERT(g      != NULL);
	ASSERT(idx    != NULL);
	ASSERT(plan   != NULL);
	ASSERT(filter != NULL);

	IndexScan *op = rm_malloc(sizeof(IndexScan));
	op->g                    =  g;
	op->n                    =  n;
	op->idx                  =  idx;
	op->iter                 =  NULL;
	op->filter               =  filter;
	op->child_record         =  NULL;
	op->unresolved_filters   =  NULL;
	op->rebuild_index_query  =  false;

	// Set our Op operations
	OpBase_Init((OpBase *)op, OPType_NODE_BY_INDEX_SCAN, "Node By Index Scan", IndexScanInit, IndexScanConsume,
				IndexScanReset, IndexScanToString, NULL, IndexScanFree, false, plan);

	op->nodeRecIdx = OpBase_Modifies((OpBase *)op, n->alias);
	return (OpBase *)op;
}

static OpResult IndexScanInit(OpBase *opBase) {
	IndexScan *op = (IndexScan *)opBase;

	if(opBase->childCount > 0) {
		// find out how many different entities are refered to 
		// within the filter tree, if number of entities equals 1
		// (current node being scanned) there's no need to re-build the index
		// query for every input record
		rax *entities = FilterTree_CollectModified(op->filter);
		op->rebuild_index_query = raxSize(entities) > 1; // this is us
		raxFree(entities);

		OpBase_UpdateConsume(opBase, IndexScanConsumeFromChild);
	}

	// resolve label ID now if it is still unknown
	if(op->n->label_id == GRAPH_UNKNOWN_LABEL) {
		GraphContext *gc = QueryCtx_GetGraphCtx();
		Schema *schema = GraphContext_GetSchema(gc, op->n->label, SCHEMA_NODE);
		ASSERT(schema != NULL);
		op->n->label_id = schema->id;
	}

	return OP_OK;
}

static inline void _UpdateRecord(IndexScan *op, Record r, EntityID node_id) {
	// Populate the Record with the graph entity data.
	Node n = GE_NEW_NODE();
	int res = Graph_GetNode(op->g, node_id, &n);
	ASSERT(res != 0);
	Record_AddNode(r, op->nodeRecIdx, n);
}

static inline bool _PassUnresolvedFilters(const IndexScan *op, Record r) {
	FT_FilterNode *unresolved_filters = op->unresolved_filters;
	if(unresolved_filters == NULL) return true; // no filters
	return FilterTree_applyFilters(unresolved_filters, r) == FILTER_PASS;
}

static Record IndexScanConsumeFromChild(OpBase *opBase) {
	IndexScan *op = (IndexScan *)opBase;
	RSIndex *rsIdx = Index_RSIndex(op->idx);
	const EntityID *nodeId = NULL;

pull_index:
	//--------------------------------------------------------------------------
	// pull from index
	//--------------------------------------------------------------------------

	if(op->iter != NULL && op->child_record != NULL) {
		while((nodeId = RediSearch_ResultsIteratorNext(op->iter, rsIdx, NULL))
				!= NULL) {
			// populate record with node
			_UpdateRecord(op, op->child_record, *nodeId);
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
		OpBase_DeleteRecord(&op->child_record);
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
		RSQNode *rs_query_node = Index_BuildQueryTree(&op->unresolved_filters,
				op->idx, filter);
		FilterTree_Free(filter);

		// create iterator
		ASSERT(rs_query_node != NULL);
		op->iter = RediSearch_GetResultsIterator(rs_query_node, rsIdx);
	} else {
		// build index query only once (first call)
		// reset it if already initialized
		if(op->iter == NULL) {
			// first call to consume, create query and iterator
			RSQNode *rs_query_node = Index_BuildQueryTree(
					&op->unresolved_filters, op->idx, op->filter);
			ASSERT(rs_query_node != NULL);
			op->iter = RediSearch_GetResultsIterator(rs_query_node, rsIdx);
		} else {
			// reset existing iterator
			RediSearch_ResultsIteratorReset(op->iter);
		}
	}

	// repull from index
	goto pull_index;
}

static Record IndexScanConsume(OpBase *opBase) {
	IndexScan *op = (IndexScan *)opBase;
	RSIndex *rsIdx = Index_RSIndex(op->idx);

	// create iterator on first call
	if(op->iter == NULL) {
		RSQNode *rs_query_node = Index_BuildQueryTree(&op->unresolved_filters,
				op->idx, op->filter);

		op->iter = RediSearch_GetResultsIterator(rs_query_node, rsIdx);
	}

	const EntityID *nodeId = NULL;

	// populate the Record with the actual node
	Record r = OpBase_CreateRecord((OpBase *)op);
	while((nodeId = RediSearch_ResultsIteratorNext(op->iter, rsIdx, NULL))
			!= NULL) {
		// populate record with node
		_UpdateRecord(op, r, *nodeId);
		// apply unresolved filters
		if(_PassUnresolvedFilters(op, r)) {
			return r;
		}
	}

	OpBase_DeleteRecord(&r);

	return NULL;
}

static OpResult IndexScanReset(OpBase *opBase) {
	IndexScan *op = (IndexScan *)opBase;

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

static void IndexScanFree(OpBase *opBase) {
	IndexScan *op = (IndexScan *)opBase;
	/* As long as this Index iterator is alive the index is
	 * read locked, if this index scan operation is part of
	 * a query which will modified this index we'll be stuck in
	 * a dead lock, as we're unable to acquire index write lock. */
	if(op->iter != NULL) {
		RediSearch_ResultsIteratorFree(op->iter);
		op->iter = NULL;
	}

	if(op->child_record != NULL) {
		OpBase_DeleteRecord(&op->child_record);
	}

	if(op->filter != NULL) {
		FilterTree_Free(op->filter);
		op->filter = NULL;
	}

	if(op->unresolved_filters != NULL) {
		FilterTree_Free(op->unresolved_filters);
		op->unresolved_filters = NULL;
	}

	if(op->n != NULL) {
		NodeScanCtx_Free(op->n);
		op->n = NULL;
	}
}

