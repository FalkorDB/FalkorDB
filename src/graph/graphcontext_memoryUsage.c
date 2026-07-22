/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "graph/graph.h"
#include "graphcontext.h"
#include "../index/index.h"
#include "../schema/schema.h"
#include "util/simple_rand.h"
#include "graph_memoryUsage.h"
#include "entities/attribute_set.h"

#include <sys/param.h>

#define MB (1 << 20)

//------------------------------------------------------------------------------
// attribute and index estimation helpers
//------------------------------------------------------------------------------

// estimate total memory usage for all entities in the datablock
static size_t _TotalAttributeMemory
(
	DataBlockIterator *it  // DataBlock to iterate
) {
	ASSERT (it != NULL) ;

	AttributeSet *set = NULL ;
	size_t memory_usage = 0 ;

	while ((set = (AttributeSet*) (DataBlockIterator_Next (it, NULL))) != NULL) {
		// entity has no attributes, skip
		if (*set != NULL) {
			memory_usage += AttributeSet_memoryUsage (*set) ;
		}
	}

	DataBlockIterator_Free (it) ;
	return memory_usage ;
}

// estimate total memory usage for all entities in the datablock
static size_t _EstimateTotalAttributeMemory
(
	DataBlock *block,     // DataBlock to iterate
	int64_t sample_size   // number of entities to sample
) {
	ASSERT (block != NULL) ;

	int64_t itemcount = DataBlock_ItemCount(block) ;
	
	if (itemcount == 0) {
		return 0;
	} else if (sample_size * 10 > itemcount) {
		// if sample is close to the total, just scan all nodes.
		return _TotalAttributeMemory(DataBlock_Scan(block)) ;
	}

	int64_t datablock_size = DataBlock_DeletedItemsCount(block) + itemcount;

	// TODO: find a better (more random) sampling method
	uint64_t r = 12345;
	AttributeSet *set = NULL ;
	size_t memory_usage = 0 ;
	int64_t i = 0 ;
	while (i < sample_size) {
		simple_rand (&r) ;
		set = (AttributeSet*) DataBlock_GetItem (block, r % datablock_size) ;

		if (set != NULL) {
			memory_usage += AttributeSet_memoryUsage (*set) ;
			++i ;
		}
		// skip deleted items
	}

	double avg = memory_usage / (double) sample_size ;
	return avg * itemcount ;
}

// returns the amortized memory consumption of a graph
// populates all MemoryUsageResult fields and converts all sizes to MB on return
// caller must hold at least the graph read lock
void GraphContext_EstimateMemoryUsage
(
	GraphContext      *gc,
	double             samples,
	MemoryUsageResult *result
) {
	ASSERT (gc      != NULL) ;
	ASSERT (samples > 0) ;
	ASSERT (result  != NULL) ;

	const Graph *g = GraphContext_GetGraph (gc) ;

	Graph_memoryUsage (g, result) ;

	//--------------------------------------------------------------------------
	// collect attribute set memory usage
	//--------------------------------------------------------------------------
	result->node_attr_sz = _EstimateTotalAttributeMemory (g->nodes, samples) ;
	result->edge_attr_sz = _EstimateTotalAttributeMemory (g->edges, samples) ;

	//--------------------------------------------------------------------------
	// collect indices memory usage
	//--------------------------------------------------------------------------

	int n_node_schema = GraphContext_SchemaCount (gc, SCHEMA_NODE) ;
	for (int i = 0 ; i < n_node_schema ; i++) {
		Schema *s = GraphContext_GetSchemaByID (gc, i, SCHEMA_NODE) ;

		if (!Schema_HasIndices (s)) continue ;

		Index active_idx  = ACTIVE_IDX (s) ;
		Index pending_idx = PENDING_IDX (s) ;

		if (active_idx  != NULL) result->indices_sz += Index_MemoryUsage (active_idx) ;
		if (pending_idx != NULL) result->indices_sz += Index_MemoryUsage (pending_idx) ;
	}

	int n_edge_schema = GraphContext_SchemaCount (gc, SCHEMA_EDGE) ;
	for (int i = 0 ; i < n_edge_schema ; i++) {
		Schema *s = GraphContext_GetSchemaByID (gc, i, SCHEMA_EDGE) ;

		if (!Schema_HasIndices (s)) continue ;

		Index active_idx  = ACTIVE_IDX (s) ;
		Index pending_idx = PENDING_IDX (s) ;

		if (active_idx  != NULL) result->indices_sz += Index_MemoryUsage (active_idx) ;
		if (pending_idx != NULL) result->indices_sz += Index_MemoryUsage (pending_idx) ;
	}

	//--------------------------------------------------------------------------
	// sum and convert all fields from bytes to MB
	//--------------------------------------------------------------------------

	result->total_graph_sz_mb +=
		result->indices_sz             +
		result->lbl_matrices_sz        +
		result->rel_matrices_sz        +
		result->node_attr_sz           +
		result->edge_attr_sz           +
		result->node_block_storage_sz  +
		result->edge_block_storage_sz ;

	result->indices_sz             /= MB ;
	result->lbl_matrices_sz        /= MB ;
	result->rel_matrices_sz        /= MB ;
	result->node_attr_sz           /= MB ;
	result->edge_attr_sz           /= MB ;
	result->node_block_storage_sz  /= MB ;
	result->edge_block_storage_sz  /= MB ;

	result->total_graph_sz_mb      /= MB ;
}

