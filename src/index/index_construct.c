/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "index.h"

#include "../graph/tensor/tensor.h"
#include "../graph/delta_matrix/delta_matrix_iter.h"

#include <assert.h>

// index nodes in an asynchronous manner
// nodes are being indexed in batchs while the graph's read lock is held
// to avoid interfering with the DB ongoing operation after each batch of nodes
// is indexed the graph read lock is released
// alowing for write queries to be processed
//
// it is safe to run a write query which effects the index by either:
// adding/removing/updating an entity while the index is being populated
// in the "worst" case we will index that entity twice which is perfectly OK
static void _Index_PopulateNodeIndex
(
	Index idx,
	Graph *g
) {
	ASSERT(g   != NULL);
	ASSERT(idx != NULL);

	GrB_Index          rowIdx     = 0;
	int                indexed    = 0;      // #entities in current batch
	int                batch_size = 10000;  // max #entities to index in one go
	Delta_MatrixTupleIter it      = {0};

	while(true) {
		// lock graph for reading
		Graph_AcquireReadLock(g);

		// index state changed, abort indexing
		// this can happen if for example the following sequance is issued:
		// 1. CREATE INDEX FOR (n:Person) ON (n.age)
		// 2. CREATE INDEX FOR (n:Person) ON (n.height)
		if(Index_PendingChanges(idx) > 1) {
			break;
		}

		// reset number of indexed nodes in batch
		indexed = 0;

		// fetch label matrix
		const Delta_Matrix m = Graph_GetLabelMatrix(g, Index_GetLabelID(idx));
		ASSERT(m != NULL);

		//----------------------------------------------------------------------
		// resume scanning from rowIdx
		//----------------------------------------------------------------------

		GrB_Info info;
		info = Delta_MatrixTupleIter_attach(&it, m);
		ASSERT(info == GrB_SUCCESS);
		info = Delta_MatrixTupleIter_iterate_range(&it, rowIdx, UINT64_MAX);
		ASSERT(info == GrB_SUCCESS);

		//----------------------------------------------------------------------
		// batch index nodes
		//----------------------------------------------------------------------

		EntityID id;
		while(indexed < batch_size &&
			  Delta_MatrixTupleIter_next_BOOL(&it, &id, NULL, NULL) == GrB_SUCCESS)
		{
			Node n;
			Graph_GetNode(g, id, &n);
			Index_IndexNode(idx, &n);
			indexed++;
		}

		//----------------------------------------------------------------------
		// done with current batch
		//----------------------------------------------------------------------

		if(indexed != batch_size) {
			// iterator depleted, no more nodes to index
			break;
		} else {
			// release read lock
			Graph_ReleaseLock(g);

			// finished current batch
			Delta_MatrixTupleIter_detach(&it);

			// continue next batch from row id+1
			// this is true because we're iterating over a diagonal matrix
			rowIdx = id + 1;
		}
	}

	// release read lock
	Graph_ReleaseLock(g);
	Delta_MatrixTupleIter_detach(&it);
}

// index edges in an asynchronous manner
// edges are being indexed in batchs while the graph's read lock is held
// to avoid interfering with the DB ongoing operation after each batch of edges
// is indexed the graph read lock is released
// alowing for write queries to be processed
//
// it is safe to run a write query which effects the index by either:
// adding/removing/updating an entity while the index is being populated
// in the "worst" case we will index that entity twice which is perfectly OK
static void _Index_PopulateEdgeIndex
(
	Index idx,
	Graph *g
) {
	ASSERT(g   != NULL);
	ASSERT(idx != NULL);

	bool  info;
	EntityID  src_id       = 0;                      // current processed row idx
	EntityID  dest_id      = 0;                      // current processed column idx
	EntityID  edge_id      = 0;                      // current processed edge id
	EntityID  prev_src_id  = INVALID_ENTITY_ID;      // last processed row idx
	EntityID  prev_dest_id = INVALID_ENTITY_ID;      // last processed column idx
	int       indexed      = 0;                      // number of entities indexed in current batch
	int       schema_id    = Index_GetLabelID(idx);  // index relationship type ID
	int       batch_size   = 1000;                   // max number of entities to index in one go
	TensorIterator it      = {0};                    // relation matrix iterator

	while(true) {
		// lock graph for reading
		Graph_AcquireReadLock(g);

		// index state changed, abort indexing
		// this can happen if for example the following sequance is issued:
		// 1. CREATE INDEX FOR (:Person)-[e:WORKS]-(:Company) ON (e.since)
		// 2. CREATE INDEX FOR (:Person)-[e:WORKS]-(:Company) ON (e.title)
		if(Index_PendingChanges(idx) > 1) {
			break;
		}

		// reset number of indexed edges in batch
		indexed = 0;

		// fetch relation matrix
		Tensor R = Graph_GetRelationMatrix(g, Index_GetLabelID(idx), false);

		//----------------------------------------------------------------------
		// resume scanning from previous row/col indices
		//----------------------------------------------------------------------

		TensorIterator_ScanRange(&it, R, src_id, UINT64_MAX, false);

		// skip previously indexed edges
		while((info =
				TensorIterator_next(&it, &src_id, &dest_id, &edge_id, NULL)) &&
				src_id == prev_src_id &&
				dest_id < prev_dest_id);

		// process only if iterator is on an active entry
		if(!info) {
			break;
		}

		//----------------------------------------------------------------------
		// batch index edges
		//----------------------------------------------------------------------

		do {
			Edge e;
			e.src_id     = src_id;
			e.dest_id    = dest_id;
			e.relationID = Index_GetLabelID(idx);

			Graph_GetEdge(g, edge_id, &e);
			Index_IndexEdge(idx, &e);

			if(prev_src_id != src_id || prev_dest_id != dest_id) {
				indexed++;
			}
			prev_src_id  = src_id;
			prev_dest_id = dest_id;
		} while((indexed < batch_size || (prev_src_id == src_id && prev_dest_id == dest_id)) &&
			  TensorIterator_next(&it, &src_id, &dest_id, &edge_id, NULL));

		//----------------------------------------------------------------------
		// done with current batch
		//----------------------------------------------------------------------

		if(indexed != batch_size) {
			// iterator depleted, no more edges to index
			break;
		} else {
			// finished current batch
			// release read lock
			Graph_ReleaseLock(g);
		}
	}

	// release read lock
	Graph_ReleaseLock(g);
}

// constructs index
void Index_Populate
(
	Index idx,
	Graph *g
) {
	ASSERT(g != NULL);
	ASSERT(idx != NULL);
	ASSERT(!Index_Enabled(idx));  // index should have pending changes

	//--------------------------------------------------------------------------
	// populate index
	//--------------------------------------------------------------------------

	if(Index_GraphEntityType(idx) == GETYPE_NODE) {
		_Index_PopulateNodeIndex(idx, g);
	} else {
		_Index_PopulateEdgeIndex(idx, g);
	}
}

