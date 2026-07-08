/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "proc_sp_paths.h"
#include "../value.h"
#include "../util/arr.h"
#include "../util/heap.h"
#include "../util/dict.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../errors/errors.h"
#include "../graph/graphcontext.h"
#include "../datatypes/datatypes.h"

#include <float.h>

#define UNBOUNDED_PATH_LENGTH INT64_MAX -1

// MATCH (n:L {v: 1}), (m:L {v: 5})
// CALL algo.SPpaths({sourceNode: n,
//					  targetNode: m,
//					  relTypes: ['E'], 
//					  maxLen: 3,
//					  weightProp: 'weight',
//					  costProp: 'cost',
//					  maxCost: 4,
//					  pathCount: 2}) YIELD path, pathWeight, pathCost
// RETURN path, pathWeight, pathCost

typedef struct {
	Path *path;      // path
	double weight;   // path weight
	double cost;     // path cost
} WeightedPath;

typedef struct {
	Node node;
	Edge edge;
} LevelConnection;

typedef struct {
	LevelConnection **levels;    // nodes reached at depth i, and edges leading to them
	Path *path;                  // current path
	Graph *g;                    // graph to traverse
	Edge *neighbors;             // reusable buffer of edges along the current path
	int *relationIDs;            // edge type(s) to traverse
	Tensor *relationMatrices;    // relation matrix per relationIDs entry, synced once up front
	int relationCount;           // length of relationIDs
	GRAPH_EDGE_DIR dir;          // traverse direction
	int64_t minLen;              // path minimum length
	int64_t maxLen;              // path max length
	Node src;                    // source node
	Node *dst;                   // destination node, defaults to NULL in case of general all paths execution
	AttributeID weight_prop;     // weight attribute id
	AttributeID cost_prop;       // cost attribuite id
	double max_cost;             // maximum cost of path
	uint64_t path_count;         // number of paths to return
	union {
		WeightedPath single;     // path_count == 1
		heap_t *heap;            // in case path_count > 1
		WeightedPath *array;     // path_count == 0 return all minimum result
	};                           // path collection
	SIValue output[3];           // result returned
	SIValue *yield_path;         // yield path
	SIValue *yield_path_weight;  // yield path weight
	SIValue *yield_path_cost;    // yield path cost
} SinglePairCtx;

// free SinglePairCtx
static void SinglePairCtx_Free
(
	SinglePairCtx *ctx
) {
	if(ctx == NULL) return;

	uint32_t levelsCount = arr_len(ctx->levels);
	for(int i = 0; i < levelsCount; i++) {
		arr_free(ctx->levels[i]);
	}

	if(ctx->path)             Path_Free(ctx->path);
	if(ctx->levels)           arr_free(ctx->levels);
	if(ctx->neighbors)        arr_free(ctx->neighbors);
	if(ctx->relationIDs)      arr_free(ctx->relationIDs);
	if(ctx->relationMatrices) arr_free(ctx->relationMatrices);

	if(ctx->path_count == 0 && ctx->array != NULL) {
		arr_free(ctx->array);
	} else if(ctx->path_count > 1 && ctx->heap != NULL) {
		Heap_free(ctx->heap);
	}

	rm_free(ctx);
}

// initialize returned values pointers
static void _process_yield
(
	SinglePairCtx *ctx,
	const char **yield
) {
	ctx->yield_path        = NULL ;
	ctx->yield_path_weight = NULL ;
	ctx->yield_path_cost   = NULL ;

	int idx = 0;
	for(uint i = 0; i < arr_len(yield); i++) {
		if(strcasecmp("path", yield[i]) == 0) {
			ctx->yield_path = ctx->output + idx;
			idx++;
			continue;
		}

		if(strcasecmp("pathWeight", yield[i]) == 0) {
			ctx->yield_path_weight = ctx->output + idx;
			idx++;
			continue;
		}

		if(strcasecmp("pathCost", yield[i]) == 0) {
			ctx->yield_path_cost = ctx->output + idx;
			idx++;
			continue;
		}
	}
}

// make sure context level array have 'cap' available entries.
static void _SinglePairCtx_EnsureLevelArrayCap
(
	SinglePairCtx *ctx,
	uint level,
	uint cap
) {
	uint len = arr_len(ctx->levels);
	if(level < len) {
		LevelConnection *current = ctx->levels[level];
		ctx->levels[level] = arr_ensure_cap(current, arr_len(current) + cap);
		return;
	}

	ASSERT(level == len);
	arr_append(ctx->levels, arr_new(LevelConnection, cap));
}

// append given 'node' to given 'level' array.
static void _SinglePairCtx_AddConnectionToLevel
(
	SinglePairCtx *ctx,
	uint level,
	Node *node,
	Edge *edge
) {
	ASSERT(level < arr_len(ctx->levels));
	LevelConnection connection;
	connection.node = *node;
	if(edge) connection.edge = *edge;
	arr_append(ctx->levels[level], connection);
}

static void SinglePairCtx_New
(
	SinglePairCtx *ctx,
	Node *src,
	Node *dst,
	Graph *g,
	int *relationIDs,
	int relationCount,
	GRAPH_EDGE_DIR dir,
	int64_t minLen,
	int64_t maxLen
) {
	ASSERT (src != NULL) ;

	ctx->g             = g ;
	ctx->dir           = dir ;
	ctx->minLen        = minLen + 1 ;
	ctx->maxLen        = maxLen + 1 ;
	ctx->relationIDs   = relationIDs ;
	ctx->relationCount = relationCount ;
	ctx->levels        = arr_new (LevelConnection *, 1) ;
	ctx->path          = Path_New (1) ;
	ctx->neighbors     = arr_new (Edge, 32) ;
	ctx->src           = *src ;
	ctx->dst           = dst ;

	// resolve and synchronize each relation's matrix once, up front, instead
	// of on every neighbor-expansion call during traversal: the procedure
	// runs under the graph's read lock for its entire lifetime, so the
	// matrices are guaranteed stable for as long as they're cached here
	ctx->relationMatrices = arr_new(Tensor, relationCount);
	for(int i = 0; i < relationCount; i++) {
		Tensor R = Graph_GetRelationMatrix(g, relationIDs[i], false);
		arr_append(ctx->relationMatrices, R);
	}

	_SinglePairCtx_EnsureLevelArrayCap(ctx, 0, 1);
	_SinglePairCtx_AddConnectionToLevel(ctx, 0, src, NULL);
}

// validate config map and initialize SinglePairCtx
static ProcedureResult validate_config
(
	SIValue config,
	SinglePairCtx *ctx
) {
	SIValue start;                // start node
	SIValue end;                  // end node
	SIValue relationships;        // relationship types allowed
	SIValue dir;                  // direction
	SIValue max_length;           // max traverse length
	SIValue weight_prop;          // weight attribute name
	SIValue cost_prop;            // cost attribute name
	SIValue max_cost;             // maximum cost
	SIValue path_count;           // # of paths to return
	
	bool start_exists         = MAP_GET (config, "sourceNode",   start) ;
	bool end_exists           = MAP_GET (config, "targetNode",   end) ;
	bool relationships_exists = MAP_GET (config, "relTypes",     relationships) ;
	bool dir_exists           = MAP_GET (config, "relDirection", dir) ;
	bool max_length_exists    = MAP_GET (config, "maxLen",       max_length) ;
	bool weight_prop_exists   = MAP_GET (config, "weightProp",   weight_prop) ;
	bool cost_prop_exists     = MAP_GET (config, "costProp",     cost_prop) ;
	bool max_cost_exists      = MAP_GET (config, "maxCost",      max_cost) ;
	bool path_count_exists    = MAP_GET (config, "pathCount",    path_count) ;

	if(!start_exists || !end_exists) {
		ErrorCtx_SetError(EMSG_SPPATH_REQUIRED);
		return false;
	}
	if(SI_TYPE(start) != T_NODE || SI_TYPE(end) != T_NODE) {
		ErrorCtx_SetError(EMSG_SPPATH_INVALID_TYPE);
		return false;
	}

	GRAPH_EDGE_DIR direction = GRAPH_EDGE_DIR_OUTGOING;
	if(dir_exists) {
		if(!(SI_TYPE(dir) & T_STRING)) {
			ErrorCtx_SetError(EMSG_REL_DIRECTION);
			return false;
		}
		if(strcasecmp(dir.stringval, "incoming") == 0) {
			direction = GRAPH_EDGE_DIR_INCOMING;
		} else if(strcasecmp(dir.stringval, "outgoing") == 0) {
			direction = GRAPH_EDGE_DIR_OUTGOING;
		} else if(strcasecmp(dir.stringval, "both") == 0) {
			direction = GRAPH_EDGE_DIR_BOTH;
		} else {
			ErrorCtx_SetError(EMSG_REL_DIRECTION);
			return false;
		}
	}

	int64_t max_length_val = UNBOUNDED_PATH_LENGTH ;
	if (max_length_exists) {
		if (SI_TYPE (max_length) != T_INT64) {
			ErrorCtx_SetError (EMSG_MUST_BE, "maxLen", "integer") ;
			return false ;
		}
		max_length_val = SI_GET_NUMERIC (max_length) ;
	}

	GraphContext *gc = QueryCtx_GetGraphCtx();
	Graph *g = QueryCtx_GetGraph();
	int *types = NULL;
	uint types_count = 0;
	if(relationships_exists) {
		if(SI_TYPE(relationships) != T_ARRAY || 
			!SIArray_AllOfType(relationships, T_STRING)) {
			ErrorCtx_SetError(EMSG_MUST_BE, "relTypes", "array of strings");
			return false;
		}
		types_count = SIArray_Length(relationships);
		if(types_count > 0) {
			types = arr_new(int, types_count);
			for (uint i = 0; i < types_count; i++) {
				SIValue rel = SIArray_Get(relationships, i);
				const char *type = rel.stringval;
				Schema *s = GraphContext_GetSchema(gc, type, SCHEMA_EDGE);
				if(s == NULL) continue;
				arr_append(types, Schema_GetID(s));
			}
			types_count = arr_len(types);
		}
	} else {
		// no relTypes specified: traverse every relation type. expand to
		// concrete relation ids up front (rather than passing the
		// GRAPH_NO_RELATION wildcard through) so each one can be resolved
		// to a matrix and cached once in SinglePairCtx_New below.
		types_count = Graph_RelationTypeCount(g);
		types = arr_new(int, types_count);
		for(uint i = 0; i < types_count; i++) {
			arr_append(types, (int)i);
		}
	}

	SinglePairCtx_New(ctx, (Node *)start.ptrval, (Node *)end.ptrval, g, types,
		types_count, direction, 1, max_length_val);

	ctx->weight_prop = ATTRIBUTE_ID_NONE;
	ctx->cost_prop = ATTRIBUTE_ID_NONE;
	ctx->max_cost = DBL_MAX;
	ctx->path_count = 1;
	
	if(weight_prop_exists) {
		if(!(SI_TYPE(weight_prop) & T_STRING)) {
			ErrorCtx_SetError(EMSG_MUST_BE, "weightProp", "string");
			return false;
		}
		ctx->weight_prop = GraphContext_GetAttributeID(gc, weight_prop.stringval);
	}

	if(cost_prop_exists) {
		if(!(SI_TYPE(cost_prop) & T_STRING)) {
			ErrorCtx_SetError(EMSG_MUST_BE, "costProp", "string");
			return false;
		}
		ctx->cost_prop = GraphContext_GetAttributeID(gc, cost_prop.stringval);
	}

	if(max_cost_exists) {
		if(SI_TYPE(max_cost) != T_INT64 && SI_TYPE(max_cost) != T_DOUBLE) {
			ErrorCtx_SetError(EMSG_MUST_BE, "maxCost", "numeric");
			return false;
		}
		ctx->max_cost = SI_GET_NUMERIC(max_cost);
	}

	if(path_count_exists) {
		if(SI_TYPE(path_count) != T_INT64) {
				ErrorCtx_SetError(EMSG_MUST_BE, "pathCount", "integer");
				return false;
		}
		if(path_count.longval < 0) {
				ErrorCtx_SetError(EMSG_MUST_BE_NON_NEGATIVE, "pathCount");
				return false;
		}
		ctx->path_count = SI_GET_NUMERIC(path_count);
	}

	return true;
}

// check to see if context levels array has entries at position 'level'.
static bool _SinglePairCtx_LevelNotEmpty
(
	const SinglePairCtx *ctx,
	uint level
) {
	return (level < arr_len(ctx->levels) && arr_len(ctx->levels[level]) > 0);
}

static void addOutgoingNeighbors
(
	SinglePairCtx *ctx,
	LevelConnection *frontier,
	uint32_t depth
) {
	EntityID frontierId = INVALID_ENTITY_ID;
	if(depth > 1) frontierId = ENTITY_GET_ID(&frontier->edge);

	// Get frontier neighbors.
	for(int i = 0; i < ctx->relationCount; i++) {
		Graph_GetNodeEdgesFromMatrix(ctx->g, &frontier->node, GRAPH_EDGE_DIR_OUTGOING,
				ctx->relationMatrices[i], ctx->relationIDs[i], &ctx->neighbors);
	}

	// Add unvisited neighbors to next level.
	uint32_t neighborsCount = arr_len(ctx->neighbors);

	_SinglePairCtx_EnsureLevelArrayCap(ctx, depth, neighborsCount);
	for(uint32_t i = 0; i < neighborsCount; i++) {
		// Don't follow the frontier edge again.
		if(frontierId == ENTITY_GET_ID(ctx->neighbors + i)) continue;
		// Set the neighbor by following the edge in the correct directoin.
		Node neighbor = GE_NEW_NODE();
		Graph_GetNode(ctx->g, Edge_GetDestNodeID(ctx->neighbors + i), &neighbor);
		// Add the node and edge to the frontier.
		_SinglePairCtx_AddConnectionToLevel(ctx, depth, &neighbor, (ctx->neighbors + i));
	}
	arr_clear(ctx->neighbors);
}

static void addIncomingNeighbors
(
	SinglePairCtx *ctx,
	LevelConnection *frontier,
	uint32_t depth
) {
	EntityID frontierId = INVALID_ENTITY_ID;
	if(depth > 1) frontierId = ENTITY_GET_ID(&frontier->edge);

	// Get frontier neighbors.
	for(int i = 0; i < ctx->relationCount; i++) {
		Graph_GetNodeEdgesFromMatrix(ctx->g, &frontier->node, GRAPH_EDGE_DIR_INCOMING,
				ctx->relationMatrices[i], ctx->relationIDs[i], &ctx->neighbors);
	}

	// Add unvisited neighbors to next level.
	uint32_t neighborsCount = arr_len(ctx->neighbors);

	_SinglePairCtx_EnsureLevelArrayCap(ctx, depth, neighborsCount);
	for(uint32_t i = 0; i < neighborsCount; i++) {
		// Don't follow the frontier edge again.
		if(frontierId == ENTITY_GET_ID(ctx->neighbors + i)) continue;
		// Set the neighbor by following the edge in the correct directoin.
		Node neighbor = GE_NEW_NODE();
		Graph_GetNode(ctx->g, Edge_GetSrcNodeID(ctx->neighbors + i), &neighbor);
		// Add the node and edge to the frontier.
		_SinglePairCtx_AddConnectionToLevel(ctx, depth, &neighbor, (ctx->neighbors + i));
	}
	arr_clear(ctx->neighbors);
}

// traverse from the frontier node in the specified direction and add all encountered nodes and edges.
static void addNeighbors
(
	SinglePairCtx *ctx,
	LevelConnection *frontier,
	uint32_t depth,
	GRAPH_EDGE_DIR dir
) {
	switch(dir) {
		case GRAPH_EDGE_DIR_OUTGOING:
			addOutgoingNeighbors(ctx, frontier, depth);
			break;
		case GRAPH_EDGE_DIR_INCOMING:
			addIncomingNeighbors(ctx, frontier, depth);
			break;
		case GRAPH_EDGE_DIR_BOTH:
			addIncomingNeighbors(ctx, frontier, depth);
			addOutgoingNeighbors(ctx, frontier, depth);
			break;
		default:
			ASSERT(false && "encountered unexpected traversal direction in AllPaths");
			break;
	}
}

// get numeric attribute value of an entity otherwise return default value
static inline SIValue _get_value_or_default
(
	GraphEntity *ge,
	AttributeID id,
	SIValue default_value
) {
	SIValue v ;

	if (!GraphEntity_GetProperty (ge, id, &v)) {
		return default_value ;
	}

	if (SI_TYPE (v) & SI_NUMERIC) {
		return v ;
	}

	return default_value ;
}

// predecessor of a node discovered by the BFS pre-pass in `_find_bound_path`
typedef struct {
	NodeID parent;  // node from which this node was first reached
	Edge edge;      // edge connecting parent -> this node
} BoundParentRecord;

// quickly search for *some* concrete path from src to dst honoring
// relTypes/relDirection/maxLen (weight and cost are ignored while
// traversing) via a plain BFS.
//
// the result is used only to seed a tight initial bound for the exhaustive
// weighted search below; because it is a real, concrete path (not an
// estimate) its weight is always safe to use as an upper bound.
//
// *exists is set to false when no structural path can be found within
// maxLen hops, in which case no path can exist regardless of maxCost
// either, and the caller can skip the exhaustive search entirely.
static void _find_bound_path
(
	SinglePairCtx *ctx,
	bool *exists,
	double *out_weight,
	double *out_cost
) {
	*exists = false;

	NodeID src_id = ENTITY_GET_ID (&ctx->src) ;
	NodeID dst_id = ENTITY_GET_ID (ctx->dst)  ;

	if (src_id == dst_id) {
		return ;
	}

	// node -> 1-based index into 'records' (0 is reserved, unused, as
	// HashTableFetchValue returns NULL for it same as for a missing key;
	// src itself is inserted with value 0 and is never looked up, since
	// backtracking stops as soon as it's reached).
	// records is a single growable arena: one malloc/realloc for the whole
	// pre-pass instead of one rm_malloc per discovered node.
	dict *parents = HashTableCreate(&def_dt);
	BoundParentRecord *records = arr_new(BoundParentRecord, 64);

	NodeID *frontier = arr_new(NodeID, 1);
	NodeID *next     = arr_new(NodeID, 0);
	arr_append(frontier, src_id);

	HashTableAdd(parents, (void *)(uintptr_t)src_id, (void *)(uintptr_t)0);

	bool found = false;
	uint32_t max_hops = ctx->maxLen - 1;

	// direction(s) to expand on; depends only on ctx->dir, so compute once
	// up front rather than per node/hop.
	GRAPH_EDGE_DIR dirs[2];
	int ndirs = 0;
	if(ctx->dir == GRAPH_EDGE_DIR_OUTGOING || ctx->dir == GRAPH_EDGE_DIR_BOTH) {
		dirs[ndirs++] = GRAPH_EDGE_DIR_OUTGOING;
	}
	if(ctx->dir == GRAPH_EDGE_DIR_INCOMING || ctx->dir == GRAPH_EDGE_DIR_BOTH) {
		dirs[ndirs++] = GRAPH_EDGE_DIR_INCOMING;
	}

	for (uint32_t hop = 1; !found && hop <= max_hops && arr_len(frontier) > 0; hop++) {
		for(uint32_t i = 0; !found && i < arr_len(frontier); i++) {
			NodeID cur = frontier[i];
			Node curNode = GE_NEW_NODE();
			Graph_GetNode(ctx->g, cur, &curNode);

			for(int d = 0; !found && d < ndirs; d++) {
				for(int r = 0; r < ctx->relationCount; r++) {
					Graph_GetNodeEdgesFromMatrix(ctx->g, &curNode, dirs[d],
							ctx->relationMatrices[r], ctx->relationIDs[r], &ctx->neighbors);
				}

				uint32_t n = arr_len(ctx->neighbors);
				for(uint32_t j = 0; j < n; j++) {
					NodeID nid = (dirs[d] == GRAPH_EDGE_DIR_OUTGOING)
						? Edge_GetDestNodeID(ctx->neighbors + j)
						: Edge_GetSrcNodeID(ctx->neighbors + j);

					// single lookup that both checks membership and, if
					// absent, reserves the slot -- avoids a separate
					// find + add pair of hash lookups per candidate.
					dictEntry *existing;
					dictEntry *entry = HashTableAddRaw(parents, (void *)(uintptr_t)nid, &existing);
					if(entry == NULL) continue;  // already visited

					BoundParentRecord rec = { .parent = cur, .edge = ctx->neighbors[j] };
					arr_append(records, rec);
					HashTableSetVal(parents, entry, (void *)(uintptr_t)arr_len(records));
					arr_append(next, nid);

					if(nid == dst_id) {
						found = true;
						break;
					}
				}

				arr_clear(ctx->neighbors);
			}
		}

		NodeID *tmp = frontier;
		frontier = next;
		next = tmp;
		arr_clear(next);
	}

	arr_free(frontier);
	arr_free(next);

	if(found) {
		*exists = true;

		double weight = 0;
		double cost   = 0;
		NodeID cur = dst_id;
		while(cur != src_id) {
			uintptr_t idx = (uintptr_t)HashTableFetchValue(parents, (void *)(uintptr_t)cur);
			ASSERT(idx != 0);
			BoundParentRecord *rec = records + (idx - 1);

			SIValue c = _get_value_or_default((GraphEntity *)&rec->edge, ctx->cost_prop,   SI_LongVal(1));
			SIValue w = _get_value_or_default((GraphEntity *)&rec->edge, ctx->weight_prop, SI_LongVal(1));
			cost   += SI_GET_NUMERIC(c);
			weight += SI_GET_NUMERIC(w);

			cur = rec->parent;
		}

		*out_weight = weight;
		*out_cost   = cost;
	}

	arr_free (records) ;
	HashTableRelease (parents) ;
}

// per-node label used by the Dijkstra fast path below
typedef struct {
	NodeID parent;    // predecessor in the shortest-path tree
	Edge   edge;      // edge connecting parent -> this node
	double weight;    // current best known weight to reach this node
	bool   finalized; // true once popped from the heap with its optimal weight
} DijkstraLabel;

// heap entry: a candidate (node, weight) pair waiting to be finalized.
// duplicate/stale entries for the same node are allowed (lazy deletion);
// they're skipped at pop time via DijkstraLabel.finalized.
typedef struct {
	NodeID node;
	double weight;  // weight at the time this entry was queued (heap key)
} DijkstraItem;

// Heap_* is a max-heap by 'cmp' (see path_cmp above); invert the comparison
// so it behaves as a min-heap ordered by ascending weight.
static int _dijkstra_cmp
(
	const void *a,
	const void *b,
	void *udata
) {
	const DijkstraItem *da = a ;
	const DijkstraItem *db = b ;
	return (da->weight < db->weight) - (da->weight > db->weight) ;
}

// exact single-shortest-path search via Dijkstra (label-setting, one
// best-known weight per node, each node finalized exactly once).
//
// only valid when there is no maxCost constraint: with cost unconstrained,
// weight alone determines optimality, so classic per-node dedup applies
// and this always terminates in O((V+E) log V) -- unlike the DFS
// enumeration below, which can blow up combinatorially on graphs with many
// similar-weight alternative routes (e.g. dense/mesh-like road networks).
//
// ASSUMES weightProp is non-negative for every edge. Dijkstra's
// "finalize once, never revisit" invariant is unsound with negative
// weights (a node reached later via a heavier edge can hold a negative
// edge that retroactively beats an already-finalized node), and this is
// NOT detected or guarded against here: making that safe would require
// giving up the early-termination-at-dst optimization (running to full
// completion over the whole reachable component instead), which was
// judged not worth it given weightProp values are expected to represent
// real, non-negative quantities (distance, time, cost) in practice. if
// negative weights are ever a real requirement, this function must not
// be used as-is.
static void SPpaths_dijkstra_single
(
	SinglePairCtx *ctx
) {
	NodeID src_id = ENTITY_GET_ID (&ctx->src) ;
	NodeID dst_id = ENTITY_GET_ID (ctx->dst) ;

	// 'labels' holds one DijkstraLabel per node ever discovered (tentative
	// or finalized best-known distance, its parent and connecting edge).
	// 'label_idx' maps a node id to its 1-based slot in 'labels' (0 means
	// "not yet discovered"), since NodeIDs aren't dense/small enough to
	// index 'labels' directly.
	// 'heap' is the Dijkstra priority queue: pending (node, weight)
	// candidates ordered so the next Heap_poll always returns the
	// smallest-weight candidate discovered so far.
	dict *label_idx       = HashTableCreate (&def_dt);
	DijkstraLabel *labels = arr_new (DijkstraLabel, 64) ;
	heap_t *heap          = Heap_new (_dijkstra_cmp, NULL) ;

	// build the list of edge directions to expand through when scanning a
	// node's neighbors: OUTGOING, INCOMING, or both, per the query's
	// requested traversal direction.
	int ndirs = 0;
	GRAPH_EDGE_DIR dirs [2] ;
	if (ctx->dir == GRAPH_EDGE_DIR_OUTGOING || ctx->dir == GRAPH_EDGE_DIR_BOTH) {
		dirs [ndirs++] = GRAPH_EDGE_DIR_OUTGOING ;
	}
	if (ctx->dir == GRAPH_EDGE_DIR_INCOMING || ctx->dir == GRAPH_EDGE_DIR_BOTH) {
		dirs [ndirs++] = GRAPH_EDGE_DIR_INCOMING ;
	}

	// initialization: seed the source node with distance 0 and no parent
	// (it parents itself, which also makes the path-reconstruction loop's
	// "cur != src_id" stop condition correct). every other node is
	// implicitly at distance +inf until first discovered below.
	DijkstraLabel src_label =
		{ .parent = src_id, .weight = 0, .finalized = false } ;

	arr_append (labels, src_label) ;
	HashTableAdd (label_idx, (void *)(uintptr_t)src_id,
			(void *)(uintptr_t)arr_len (labels)) ;

	// push the source onto the priority queue so the main loop below has
	// somewhere to start.
	DijkstraItem *seed = rm_malloc (sizeof (DijkstraItem)) ;
	seed->node   = src_id ;
	seed->weight = 0 ;

	Heap_offer (&heap, seed) ;

	bool found = false ;

	// main Dijkstra loop: repeatedly extract the not-yet-finalized node
	// with the smallest tentative distance and finalize it -- that
	// distance is now guaranteed optimal, since all edge weights are
	// non-negative and every unexplored candidate is at least as large.
	// stops either when dst is finalized (found) or the heap empties
	// (dst unreachable from src).
	while (!found) {
		// extract the minimum-weight candidate. this may be a stale
		// duplicate left over from a relaxation performed after this
		// entry was queued (see the lazy-deletion note on DijkstraItem);
		// staleness is detected below via the label's 'finalized' flag
		// rather than by removing superseded heap entries in place.
		DijkstraItem *item = Heap_poll (heap) ;
		if (item == NULL) {
			break ;  // heap exhausted: dst is unreachable
		}

		NodeID cur = item->node ;
		rm_free (item) ;

		uintptr_t cur_idx =
			(uintptr_t)HashTableFetchValue (label_idx, (void *)(uintptr_t)cur) ;

		ASSERT (cur_idx != 0) ;
		if (labels [cur_idx - 1].finalized) {
			continue ;  // stale duplicate entry
		}

		// finalize 'cur': its current label weight is its true shortest
		// distance from src and will never be improved again (label
		// setting -- each node is finalized exactly once).
		labels [cur_idx - 1].finalized = true ;

		// dst just got finalized, its shortest path is settled: stop
		// early instead of exploring the rest of the reachable graph.
		if (cur == dst_id) {
			found = true ;
			break ;
		}

		double cur_weight = labels [cur_idx - 1].weight ;

		Node curNode = GE_NEW_NODE () ;
		Graph_GetNode (ctx->g, cur, &curNode) ;

		// relaxation step: examine every edge leaving (or entering, per
		// 'dirs') 'cur', across every relationship type the query allows,
		// and try to improve each neighbor's tentative distance through
		// 'cur'.
		for (int d = 0; d < ndirs; d++) {
			for (int r = 0; r < ctx->relationCount; r++) {
				Graph_GetNodeEdgesFromMatrix (ctx->g, &curNode, dirs [d],
						ctx->relationMatrices [r], ctx->relationIDs [r], &ctx->neighbors) ;
			}

			uint32_t n = arr_len (ctx->neighbors) ;
			for (uint32_t j = 0; j < n; j++) {
				Edge *e = ctx->neighbors + j ;
				NodeID nid = (dirs [d] == GRAPH_EDGE_DIR_OUTGOING)
					? Edge_GetDestNodeID (e)
					: Edge_GetSrcNodeID (e) ;

				if (nid == cur) {
					continue ;  // ignore self-loops
				}

				// candidate distance to 'nid' going through 'cur' via
				// this edge: cur's finalized distance plus the edge's
				// weight.
				// NOTE: weightProp is assumed non-negative here (see the
				// function-level comment above); a negative value would
				// silently make this search's result incorrect.
				SIValue w = _get_value_or_default ((GraphEntity *)e,
						ctx->weight_prop, SI_LongVal (1)) ;
				double new_weight = cur_weight + SI_GET_NUMERIC (w) ;

				// look up (or reserve) 'nid's slot in 'labels': HashTableAddRaw
				// returns a fresh entry (entry != NULL) the first time 'nid'
				// is seen, or NULL with 'existing' set to the prior entry if
				// 'nid' already has a label.
				dictEntry *existing ;
				dictEntry *entry = HashTableAddRaw (label_idx,
						(void *)(uintptr_t)nid, &existing) ;

				if (entry == NULL) {
					// 'nid' already labeled: this is the relaxation
					// comparison proper. skip if it's already finalized
					// (its distance is final and can't improve) or if
					// going through 'cur' isn't strictly better than what
					// it already has.
					uintptr_t idx = (uintptr_t)HashTableGetVal (existing) ;
					DijkstraLabel *nlabel = labels + (idx - 1) ;
					if (nlabel->finalized || new_weight >= nlabel->weight) {
						continue ;
					}

					// found a strictly shorter route to 'nid' through
					// 'cur': update its label in place with the new best
					// distance, parent and connecting edge.
					nlabel->edge   = *e ;
					nlabel->parent = cur ;
					nlabel->weight = new_weight ;
				} else {
					// first time 'nid' is discovered: create its label
					// with 'cur' as parent and 'new_weight' as its (so
					// far unbeaten) tentative distance.
					DijkstraLabel nlabel = { .parent = cur, .edge = *e,
						.weight = new_weight, .finalized = false } ;

					arr_append (labels, nlabel) ;
					HashTableSetVal (label_idx, entry,
							(void *)(uintptr_t)arr_len (labels)) ;
				}

				// queue (or re-queue) 'nid' at its updated tentative
				// weight. any older, now-superseded heap entry for 'nid'
				// is left in place and simply skipped later as a stale
				// duplicate once popped.
				DijkstraItem *qi = rm_malloc (sizeof (DijkstraItem)) ;
				qi->node   = nid ;
				qi->weight = new_weight ;
				Heap_offer (&heap, qi) ;
			}

			arr_clear (ctx->neighbors) ;
		}
	}

	// search is over (dst found or heap exhausted): drain and free any
	// remaining queued items before freeing the heap itself.
	DijkstraItem *leftover ;
	while ((leftover = Heap_poll (heap)) != NULL) {
		rm_free (leftover) ;
	}
	Heap_free (heap) ;

	if (!found) {
		// dst is unreachable from src: nothing to report, leave
		// ctx->single at its caller-initialized "no path" state.
		arr_free (labels) ;
		HashTableRelease (label_idx) ;
		return ;
	}

	// reconstruct the path by walking parent pointers from dst back to
	// src, one finalized label at a time, accumulating cost_prop along
	// the way (weight_prop was already accumulated into each label's
	// 'weight' during the relaxation loop above, so total_weight is just
	// read off dst's label once the walk is done).
	NodeID cur = dst_id ;
	double total_cost = 0 ;
	Path *path = Path_New (8) ;

	while (cur != src_id) {
		uintptr_t idx =
			(uintptr_t)HashTableFetchValue (label_idx, (void *)(uintptr_t)cur) ;
		ASSERT (idx != 0) ;
		DijkstraLabel *label = labels + (idx - 1) ;

		// append 'cur' and the edge that reached it from its parent; the
		// path is being built tail-first (dst towards src) and will be
		// reversed once the walk reaches src.
		Node n = GE_NEW_NODE () ;
		Graph_GetNode (ctx->g, cur, &n) ;
		Path_AppendNode (path, n) ;
		Path_AppendEdge (path, label->edge) ;

		SIValue c =
			_get_value_or_default ((GraphEntity *)&label->edge, ctx->cost_prop,
					SI_LongVal (1)) ;
		total_cost += SI_GET_NUMERIC (c) ;

		cur = label->parent ;
	}

	// walk terminated at src: append it (it has no incoming edge on this
	// path) and flip the path from dst->src order into src->dst order.
	Node srcNode = GE_NEW_NODE () ;
	Graph_GetNode (ctx->g, src_id, &srcNode) ;
	Path_AppendNode (path, srcNode) ;

	Path_Reverse (path) ;

	// dst's finalized label already holds the total shortest weight from
	// src, accumulated incrementally throughout the relaxation loop.
	uintptr_t dst_idx =
		(uintptr_t)HashTableFetchValue (label_idx, (void *)(uintptr_t)dst_id) ;
	double total_weight = labels [dst_idx - 1].weight ;

	ctx->single.path   = path ;
	ctx->single.cost   = total_cost ;
	ctx->single.weight = total_weight ;

	arr_free (labels) ;
	HashTableRelease (label_idx) ;
}

// use DFS to find all paths from src to dst tracking cost and weight
static void SPpaths_next
(
	SinglePairCtx *ctx,
	WeightedPath *p,
	double max_weight
) {
	// as long as path is not empty OR there are neighbors to traverse.
	while(Path_NodeCount(ctx->path) || _SinglePairCtx_LevelNotEmpty(ctx, 0)) {
		uint32_t depth = Path_NodeCount(ctx->path);

		// can we advance?
		if(_SinglePairCtx_LevelNotEmpty(ctx, depth)) {
			// get a new frontier.
			LevelConnection frontierConnection = arr_pop(ctx->levels[depth]);
			Node frontierNode = frontierConnection.node;

			bool frontierAlreadyOnPath =
				Path_ContainsNode(ctx->path, ENTITY_GET_ID (&frontierNode));

			// a self-loop (or duplicate relation ids) can surface the same
			// edge as a candidate more than once; reject it exactly like a
			// node cycle so it doesn't get traversed twice on one path.
			if(!frontierAlreadyOnPath && depth > 0 &&
				Path_ContainsEdge(ctx->path, ENTITY_GET_ID(&frontierConnection.edge))) {
				frontierAlreadyOnPath = true;
			}

			// don't allow cycles
			if (frontierAlreadyOnPath) {
				continue ;
			}

			// add frontier to path.
			Path_AppendNode(ctx->path, frontierNode);

			// if depth is 0 this is the source node, there is no leading edge to it.
			// For depth > 0 for each frontier node, there is a leading edge.
			if(depth > 0) {
				SIValue c = _get_value_or_default((GraphEntity *)&frontierConnection.edge, ctx->cost_prop, SI_LongVal(1));
				SIValue w = _get_value_or_default((GraphEntity *)&frontierConnection.edge, ctx->weight_prop, SI_LongVal(1));
				if(p->cost + SI_GET_NUMERIC(c) <= ctx->max_cost && p->weight + SI_GET_NUMERIC(w) <= max_weight) {
					p->cost += SI_GET_NUMERIC(c);
					p->weight += SI_GET_NUMERIC(w);
					Path_AppendEdge(ctx->path, frontierConnection.edge);
				} else {
					Path_PopNode(ctx->path);
					continue;
				}
			}

			// update path depth.
			depth++;

			// introduce neighbors only if path depth < maximum path length.
			// and frontier wasn't already expanded.
			if(depth < ctx->maxLen) {
				addNeighbors(ctx, &frontierConnection, depth, ctx->dir);
			}

			// see if we can return path.
			if(depth >= ctx->minLen && depth <= ctx->maxLen) {
				Node dst = Path_Head(ctx->path);
				if(ENTITY_GET_ID(ctx->dst) != ENTITY_GET_ID(&dst)) {
					continue;
				}
				p->path = ctx->path;
				return;
			}
		} else {
			// no way to advance, backtrack.
			Path_PopNode(ctx->path);
			if(Path_EdgeCount(ctx->path)) {
				Edge e = Path_PopEdge(ctx->path);
				SIValue c = _get_value_or_default((GraphEntity *)&e, ctx->cost_prop, SI_LongVal(1));
				SIValue w = _get_value_or_default((GraphEntity *)&e, ctx->weight_prop, SI_LongVal(1));
				p->cost -= SI_GET_NUMERIC(c);
				p->weight -= SI_GET_NUMERIC(w);
			}
		}
	}

	// couldn't find a path.
	p->path = NULL;
	return;
}

// compare path by weight, cost and path length
static int path_cmp
(
	const void *a,
	const void *b,
	void *udata
) {
	WeightedPath *da = (WeightedPath *)a;
	WeightedPath *db = (WeightedPath *)b;
	if(da->weight == db->weight) {
		if(da->cost == db->cost) {
			size_t len_a = Path_Len(da->path);
			size_t len_b = Path_Len(db->path);
			return (len_a > len_b) - (len_a < len_b);
		}
		return (da->cost > db->cost) - (da->cost < db->cost);
	}
	return (da->weight > db->weight) - (da->weight < db->weight);
}

// get all minimal paths (all paths with the same weight)
static void SPpaths_all_minimal
(
	SinglePairCtx *ctx,
	double initial_bound
) {
	// initialize array that contains the result
	ctx->array = arr_new(WeightedPath, 0);

	// get first path
	WeightedPath p = {0};
	double max_weight = initial_bound;
	SPpaths_next(ctx, &p, max_weight);

	// iterate over all paths
	while (p.path != NULL) {
		// if current path is better and the array is not empty clear it
		uint count = arr_len(ctx->array);
		if(count > 0 && p.weight < ctx->array[0].weight) {
			for (uint i = 0; i < arr_len(ctx->array); i++) {
				Path_Free(ctx->array[i].path);
			}
			arr_clear(ctx->array);
		}

		// add the path to the result array
		p.path = Path_Clone(p.path);
		arr_append(ctx->array, p);

		// update max weight
		max_weight = p.weight;

		// get next path where path weight is <= max_weight
		SPpaths_next(ctx, &p, max_weight);
	}
}

// find the single minimal weighted path
static void SPpaths_single_minimal
(
	SinglePairCtx *ctx,
	double initial_bound
) {
	// initialize the result path to worst path
	ctx->single.path   = NULL;
	ctx->single.weight = DBL_MAX;
	ctx->single.cost   = DBL_MAX;

	// get first path
	WeightedPath p = {0};
	SPpaths_next(ctx, &p, initial_bound);

	// iterate over all paths
	while (p.path != NULL) {
		// if the current path is better replace it
		if(p.weight < ctx->single.weight ||
			p.cost < ctx->single.cost ||
			(p.cost == ctx->single.cost &&
				Path_Len(p.path) < Path_Len(ctx->single.path))) {
			if(ctx->single.path != NULL) {
				Path_Free(ctx->single.path);
			}
			ctx->single.path = Path_Clone(p.path);
			ctx->single.weight = p.weight;
			ctx->single.cost = p.cost;
		}

		// get next path where path weight is <= result weight
		SPpaths_next(ctx, &p, ctx->single.weight);
	}
}

static void inline _add_path
(
	heap_t **heap,
	WeightedPath *p
) {
	WeightedPath *pp = rm_malloc(sizeof(WeightedPath));
	pp->path = Path_Clone(p->path);
	pp->weight = p->weight;
	pp->cost = p->cost;
	Heap_offer(heap, pp);
}

// find k minimal weighted path (path can have different weight)
static void SPpaths_k_minimal
(
	SinglePairCtx *ctx
) {
	// initialize heap that contains the result where top path is the highest weight
	ctx->heap = Heap_new(path_cmp, NULL);

	// get first path. unlike the single/all-minimal cases, the pre-pass
	// bound must NOT seed this search: we need to fill up to path_count
	// candidates before weight can meaningfully bound anything, since the
	// k best paths can legitimately span a range of weights above the
	// single cheapest path found by the pre-pass.
	WeightedPath p = {0};
	double max_weight = DBL_MAX;
	SPpaths_next(ctx, &p, max_weight);

	// iterate over all paths
	while (p.path != NULL && Heap_count(ctx->heap) < ctx->path_count - 1) {
		// fill the heap
		_add_path(&ctx->heap, &p);

		// get next path where path weight is <= max_weight
		SPpaths_next(ctx, &p, max_weight);
	}

	if(p.path == NULL) return;

	// fill the heap
	_add_path(&ctx->heap, &p);

	// update the max weight so we will get better paths
	WeightedPath *pp = Heap_peek(ctx->heap);
	max_weight = pp->weight;

	// get next path where path weight is <= max_weight
	SPpaths_next(ctx, &p, max_weight);

	while (p.path != NULL) {
		// if the heap is full check if the current path is better 
		// than the worst path if yes replace it
		pp = Heap_peek(ctx->heap);
		if(p.weight < pp->weight ||
			p.cost < pp->cost ||
			(p.cost == pp->cost &&
				Path_Len(p.path) < Path_Len(pp->path))) {
			Heap_poll(ctx->heap);
			Path_Free(pp->path);
			pp->path = Path_Clone(p.path);
			pp->weight = p.weight;
			pp->cost = p.cost;
			Heap_offer(&ctx->heap, pp);

			// update the max weight so we will get better paths
			pp = Heap_peek(ctx->heap);
			max_weight = pp->weight;
		}

		// get next path where path weight is <= max_weight
		SPpaths_next(ctx, &p, max_weight);
	}
}

static ProcedureResult Proc_SPpathsInvoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {
	SinglePairCtx *single_pair_ctx = rm_calloc (1, sizeof (SinglePairCtx)) ;
	if (!validate_config (args [0], single_pair_ctx)) {
		SinglePairCtx_Free (single_pair_ctx) ;
		return PROCEDURE_ERR ;
	}

	ctx->privateData = single_pair_ctx ;
	_process_yield (single_pair_ctx, yield) ;

	// fast path: a single shortest path with no maxCost constraint is
	// exactly what Dijkstra solves, in O((V+E) log V) instead of the
	// exhaustive DFS enumeration below, which can blow up combinatorially
	// on graphs with many similar-weight alternative routes. this makes
	// the bound pre-pass unnecessary too, since Dijkstra finds the exact
	// optimum (and unreachability) directly.
	// NOTE: SPpaths_dijkstra_single assumes weightProp is non-negative for
	// every edge (see its own comment for why this isn't detected/guarded).
	// src == dst is degenerate: Dijkstra trivially "finds" the source at
	// distance 0 with zero edges traversed, which would violate the
	// minLen==1 contract (a path needs at least one edge, e.g. a genuine
	// self-loop). Rather than special-casing that inside the search, just
	// don't take the fast path here and let the exhaustive DFS (which
	// already handles this correctly) run instead.
	bool src_eq_dst =
		(ENTITY_GET_ID (&single_pair_ctx->src) ==
		 ENTITY_GET_ID (single_pair_ctx->dst)) ;

	if (src_eq_dst == false                    &&
		single_pair_ctx->path_count == 1       &&
		single_pair_ctx->max_cost   == DBL_MAX &&
		single_pair_ctx->maxLen     == UNBOUNDED_PATH_LENGTH + 1) {
		SPpaths_dijkstra_single (single_pair_ctx) ;
		return PROCEDURE_OK ;
	}

	// quick pre-pass: does *any* structural path (honoring relTypes/
	// relDirection/maxLen) exist between src and dst at all? if not, no
	// path can exist regardless of maxCost either, so skip the exhaustive
	// search entirely. if one is found and it also satisfies maxCost, its
	// weight is a safe upper bound to seed the exhaustive search with.
	double bound_cost ;
	bool   bound_exists ;
	double bound_weight ;
	_find_bound_path (single_pair_ctx, &bound_exists, &bound_weight,
			&bound_cost) ;

	if (!bound_exists) {
		if(single_pair_ctx->path_count == 0) {
			single_pair_ctx->array = arr_new (WeightedPath, 0) ;
		} else if(single_pair_ctx->path_count == 1) {
			single_pair_ctx->single.path = NULL;
		} else {
			single_pair_ctx->heap = Heap_new (path_cmp, NULL) ;
		}
		return PROCEDURE_OK ;
	}

	bool cost_feasible = (bound_cost <= single_pair_ctx->max_cost) ;
	double initial_bound = cost_feasible ? bound_weight : DBL_MAX ;

	if (single_pair_ctx->path_count == 0) {
		// all-minimal wants every tie at the true minimum weight; the
		// pre-pass weight is a valid upper bound on that minimum (any tie
		// is by definition <= it), so seeding is safe here.
		SPpaths_all_minimal(single_pair_ctx, initial_bound);
	} else if(single_pair_ctx->path_count == 1) {
		SPpaths_single_minimal(single_pair_ctx, initial_bound);
	} else {
		// k-minimal needs to fill up to path_count candidates before
		// weight can meaningfully bound anything -- those candidates can
		// legitimately be heavier than the single path the pre-pass
		// found, so the bound must not be applied here.
		SPpaths_k_minimal(single_pair_ctx);
	}

	return PROCEDURE_OK;
}

static SIValue *Proc_SPpathsStep
(
	ProcedureCtx *ctx
) {
	ASSERT(ctx->privateData != NULL);
	
	SinglePairCtx *single_pair_ctx = ctx->privateData;
	WeightedPath p;

	if(single_pair_ctx->path_count == 0) {
		if(arr_len(single_pair_ctx->array) == 0) return NULL;

		p = arr_pop(single_pair_ctx->array);
	} else if(single_pair_ctx->path_count == 1) {
		p = single_pair_ctx->single;
		if(p.path == NULL) return NULL;

		single_pair_ctx->single.path = NULL;
	} else {
		WeightedPath *pp = Heap_poll(single_pair_ctx->heap);
		if(pp == NULL) return NULL;
		
		p = *pp;
		rm_free(pp);
	}
	
	if(single_pair_ctx->yield_path) {
		*single_pair_ctx->yield_path = SIPath_Wrap (&p.path) ;
	} else {
		Path_Free (p.path) ;
	}

	if(single_pair_ctx->yield_path_weight) *single_pair_ctx->yield_path_weight = SI_DoubleVal(p.weight);
	if(single_pair_ctx->yield_path_cost)   *single_pair_ctx->yield_path_cost   = SI_DoubleVal(p.cost);

	return single_pair_ctx->output;
}

static ProcedureResult Proc_SPpathsFree
(
	ProcedureCtx *ctx
) {
	ASSERT (ctx != NULL) ;

	SinglePairCtx *single_pair_ctx = ctx->privateData ;
	SinglePairCtx_Free (single_pair_ctx) ;

	return PROCEDURE_OK ;
}

ProcedureCtx *Proc_SPpathCtx (void) {
	ProcedureOutput output ;
	void *privateData = NULL ;

	ProcedureOutput *outputs = arr_newlen (ProcedureOutput, 3) ;

	outputs [0] = (ProcedureOutput) {.name = "path",       .type = T_PATH} ;
	outputs [1] = (ProcedureOutput) {.name = "pathWeight", .type = T_DOUBLE} ;
	outputs [2] = (ProcedureOutput) {.name = "pathCost",   .type = T_DOUBLE} ;

	return ProcCtxNew ("algo.SPpaths", 1, outputs, Proc_SPpathsStep,
			Proc_SPpathsInvoke, Proc_SPpathsFree, privateData, true) ;
}

