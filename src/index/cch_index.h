/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../algorithms/cch.h"
#include "../graph/graph.h"                     // Graph
#include "../graph/entities/edge.h"            // RelationID
#include "../graph/entities/attribute_set.h"   // AttributeID

#include <stdbool.h>

//------------------------------------------------------------------------------
// Customizable Contraction Hierarchies index
//------------------------------------------------------------------------------
//
// a CCHIndex is a graph-level pathfinding index: a Customizable Contraction
// Hierarchy (see algorithms/cch.h) built over the sub-graph induced by a set of
// relationship types, for a single edge-weight metric. unlike the range /
// fulltext / vector indices it is NOT scoped to a single Schema and is NOT
// backed by RediSearch -- it owns a resident CCH structure in memory. the
// hierarchy's shortcut arcs and node ranks live inside this object, never as
// user-visible graph edges or properties.
//
// identity is the pair (set of relationship types, weight attribute); a graph
// may hold several CCH indices over different metrics or relationship sets.

typedef struct _CCHIndex CCHIndex;

// pending-maintenance level accumulated from graph mutations, applied (coalesced)
// at query commit. higher levels subsume lower ones.
typedef enum {
	CCH_DIRTY_CLEAN       = 0,  // consistent with the graph; nothing to do
	CCH_DIRTY_RECUSTOMIZE = 1,  // metric changed -> re-run Phase 2 (keep order)
	CCH_DIRTY_REBUILD     = 2,  // topology changed -> re-run Phase 1 + 2
} CCHDirtyLevel;

struct _CCHIndex {
	RelationID    *rel_types;    // relationship types spanned (sorted, arr_ array)
	AttributeID    weight_attr;  // edge attribute the metric is built from
	CCH           *cch;          // resident hierarchy (Phase 1+2); NULL until built
	CCHDirtyLevel  dirty;        // pending maintenance from graph mutations
	// endpoints (node-id space) of original arcs whose weight changed since the
	// last clean -- the scope for an incremental re-customization. parallel arr_
	// arrays (dirty_u[i] -> dirty_v[i]). only meaningful while dirty ==
	// CCH_DIRTY_RECUSTOMIZE; a REBUILD supersedes and ignores them.
	NodeID        *dirty_u;
	NodeID        *dirty_v;
	// true once a pending change needs the chordal structure itself to change
	// (a new adjacency was added) -> the next flush must do a full rebuild rather
	// than a scoped recustomization
	bool           pending_rebuild;
	// count of edge deletions since the last full rebuild. deletions keep their
	// (now stale) chordal arcs, so this many-deletes counter drives the staleness
	// valve: once it crosses a threshold the structure is rebuilt to reclaim the
	// stale fill-in (also restoring elimination-order quality)
	uint64_t       stale;
};

// create a CCH index descriptor over 'rel_types' (copied then sorted) for the
// 'weight_attr' metric. the hierarchy itself is not built here -- 'cch' starts
// NULL and is populated by the build step.
CCHIndex *CCHIndex_New
(
	const RelationID *rel_types,   // relationship types the hierarchy spans
	uint              n,           // number of relationship types
	AttributeID       weight_attr  // edge weight attribute
);

// true if 'idx' is defined over exactly 'rel_types' (order-independent) and
// 'weight_attr' -- i.e. the caller is asking for this same index
bool CCHIndex_Matches
(
	const CCHIndex   *idx,         // index to test
	const RelationID *rel_types,   // requested relationship types
	uint              n,           // number of requested relationship types
	AttributeID       weight_attr  // requested weight attribute
);

// true if 'idx' spans relationship type 'r'
bool CCHIndex_CoversRelation
(
	const CCHIndex *idx,  // index to test
	RelationID      r     // relationship type
);

// the relationship types 'idx' spans (sorted); count via CCHIndex_RelTypeCount
const RelationID *CCHIndex_RelTypes
(
	const CCHIndex *idx  // index
);

// number of relationship types 'idx' spans
uint CCHIndex_RelTypeCount
(
	const CCHIndex *idx  // index
);

// the edge weight attribute 'idx' is built for
AttributeID CCHIndex_WeightAttr
(
	const CCHIndex *idx  // index
);

// true once the hierarchy has been built (Phase 1+2 materialized in 'cch')
bool CCHIndex_Built
(
	const CCHIndex *idx  // index to test
);

// build (or rebuild) the resident hierarchy from the current graph: constructs
// the weight matrix over the index's relationship types + weight attribute, then
// runs CCH preprocessing (metric-independent) and customization (metric). any
// previously-built hierarchy is freed first. synchronous.
void CCHIndex_Build
(
	CCHIndex *idx,  // index whose hierarchy to (re)build
	Graph    *g     // graph to build from
);

// re-run customization (Phase 2) over the existing elimination order and chordal
// structure, picking up new edge weights. cheaper than a full rebuild but only
// valid when the topology is unchanged (a pure metric change). requires the
// hierarchy to already be built.
void CCHIndex_Recustomize
(
	CCHIndex *idx,  // index to re-weight
	Graph    *g     // graph to read weights from
);

// raise 'idx's pending-maintenance level to at least 'level' (levels are
// monotonic: a higher level subsumes a lower one). the work is applied later,
// coalesced, at query commit.
void CCHIndex_MarkDirty
(
	CCHIndex     *idx,   // index to mark
	CCHDirtyLevel level  // maintenance level
);

// record that the original arc between node ids 'u' and 'v' changed weight,
// scoping a future incremental re-customization. also raises the dirty level to
// at least CCH_DIRTY_RECUSTOMIZE. duplicates are harmless (the consumer dedups).
void CCHIndex_MarkArcDirty
(
	CCHIndex *idx,  // index to mark
	NodeID    u,    // arc source node id
	NodeID    v     // arc destination node id
);

// record an edge addition between 'u' and 'v'. if a chordal arc already spans
// the pair it is a weight-only change (scoped recustomization); otherwise the
// addition introduces a new adjacency and the index is flagged for a full
// rebuild (the structure must grow).
void CCHIndex_MarkEdgeAdded
(
	CCHIndex *idx,  // index to mark
	NodeID    u,    // edge source node id
	NodeID    v     // edge destination node id
);

// record an edge deletion between 'u' and 'v'. the chordal arc is kept and
// re-seeded (scoped recustomization); the deletion is counted toward the
// staleness valve that periodically rebuilds to reclaim stale fill-in.
void CCHIndex_MarkEdgeDeleted
(
	CCHIndex *idx,  // index to mark
	NodeID    u,    // edge source node id
	NodeID    v     // edge destination node id
);

// clear 'idx's pending-maintenance level without applying any work
void CCHIndex_MarkClean
(
	CCHIndex *idx  // index to clear
);

// build a plain GrB_FP64 weight matrix over 'g's full NodeID space for the
// sub-graph induced by 'relTypeIDs': each edge's 'weightAtt' resolves to its
// weight (default 1), parallel/multi-type edges collapse to the cheapest. this
// is the metric matrix W consumed by CCH_Customize / CCH_ExtractShortcuts (see
// algorithms/cch.h). caller owns and frees the returned matrix.
GrB_Matrix CCH_BuildWeightMatrix
(
	Graph *g,                      // graph providing the relation matrices
	const RelationID *relTypeIDs,  // relation types forming the sub-graph
	uint relCount,                 // number of relation types
	AttributeID weightAtt          // edge attribute holding the weight
);

// total heap bytes held by the index: the descriptor, its relationship-type +
// dirty-arc arrays, and the resident hierarchy (the bulk). used by the graph
// memory-usage report, alongside the RediSearch-backed indices.
size_t CCHIndex_MemoryUsage
(
	const CCHIndex *idx  // index to measure
);

// serialize a built CCH index to 'io' (RDB): its identity (relationship types +
// weight attribute) followed by the full resident hierarchy. the index must be
// built (idx->cch != NULL).
void CCHIndex_RdbSave
(
	const CCHIndex *idx,  // index to serialize
	SerializerIO    io    // stream to write to
);

// reconstruct a fully-built CCH index from 'io' (inverse of CCHIndex_RdbSave).
// no rebuild is performed -- the hierarchy is read straight from the stream.
CCHIndex *CCHIndex_RdbLoad
(
	SerializerIO io  // stream to read from
);

void CCHIndex_Free
(
	CCHIndex *idx
);
