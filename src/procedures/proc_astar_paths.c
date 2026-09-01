/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "proc_astar_paths.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../errors/errors.h"
#include "../graph/graphcontext.h"
#include "../datatypes/datatypes.h"
#include "../algorithms/AStar.h"

// MATCH (n:L {v: 1}), (m:L {v: 5})
// CALL algo.AStar({sourceNode: n,
//                   targetNode: m,
//                   relTypes: ['E'],
//                   weightProp: 'weight',
//                   latitudeProperty: 'lat',
//                   longitudeProperty: 'lon',
//                   heuristicScale: 1.0,
//                   pathCount: 3}) YIELD path, pathWeight
// RETURN path, pathWeight
//
// heuristicScale (optional, default 1.0) scales the haversine heuristic
// (meters) into weightProp's units and must be a non-negative lower bound on
// the weight per meter for A* to stay optimal: 1.0 for a distance-in-meters
// weight, 1/max_speed for a travel-time weight (see AStar.h).

typedef struct {
	Node src;                    // source node
	Node *dst;                   // destination node, non-owning: points into
	                              // the caller-owned SIValue passed as
	                              // 'targetNode', same convention as
	                              // proc_sp_paths.c's SinglePairCtx.dst
	Graph *g;                    // graph to traverse
	RelationID *relationIDs;     // edge type(s) to traverse
	Tensor *relationMatrices;    // relation matrix per relationIDs entry
	int relationCount;           // length of relationIDs
	GRAPH_EDGE_DIR dir;          // traverse direction
	AttributeID weight_prop;     // weight attribute id
	AttributeID lat_prop;        // latitude attribute id (required)
	AttributeID lon_prop;        // longitude attribute id (required)
	double heur_scale;           // meters -> weightProp units heuristic scale
	uint64_t path_count;         // number of paths to return (>= 1)
	Path **paths;                // result paths (array_t, ascending weight)
	double *weights;             // parallel total weights (array_t)
	uint cursor;                 // index of the next path to emit
	SIValue output[2];           // result returned
	SIValue *yield_path;         // yield path
	SIValue *yield_path_weight;  // yield path weight
} AStarCtx;

// free AStarCtx
static void AStarCtx_Free
(
	AStarCtx *ctx
) {
	if (ctx == NULL) return;

	// free any result paths not yet emitted (the emitted ones were consumed by
	// SIPath_Wrap or Path_Free in the step function).
	if (ctx->paths != NULL) {
		for(uint i = ctx->cursor; i < arr_len(ctx->paths); i++) {
			Path_Free(ctx->paths[i]);
		}
		arr_free(ctx->paths);
	}
	if (ctx->weights != NULL) arr_free(ctx->weights);

	if (ctx->relationIDs)      arr_free(ctx->relationIDs);
	if (ctx->relationMatrices) arr_free(ctx->relationMatrices);

	rm_free(ctx);
}

// initialize returned values pointers
static void _process_yield
(
	AStarCtx *ctx,
	const char **yield
) {
	ctx->yield_path        = NULL;
	ctx->yield_path_weight = NULL;

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
	}
}

// validate config map and initialize AStarCtx.
// returns true on success, false on a validation error (with the error set via
// ErrorCtx). deliberately bool, not ProcedureResult: the caller uses it as a
// boolean and PROCEDURE_OK == 0 would read as "false" under that usage.
static bool validate_config
(
	SIValue config,
	AStarCtx *ctx
) {
	SIValue start;          // start node
	SIValue end;            // end node
	SIValue relationships;  // relationship types allowed
	SIValue dir;            // direction
	SIValue weight_prop;    // weight attribute name
	SIValue lat_prop;       // latitude attribute name
	SIValue lon_prop;       // longitude attribute name
	SIValue heur_scale;     // meters -> weightProp units heuristic scale
	SIValue path_count;     // # of paths to return

	bool start_exists         = MAP_GETCASEINSENSITIVE (config, "sourceNode",        start) ;
	bool end_exists           = MAP_GETCASEINSENSITIVE (config, "targetNode",        end) ;
	bool relationships_exists = MAP_GETCASEINSENSITIVE (config, "relTypes",          relationships) ;
	bool dir_exists           = MAP_GETCASEINSENSITIVE (config, "relDirection",      dir) ;
	bool weight_prop_exists   = MAP_GETCASEINSENSITIVE (config, "weightProp",        weight_prop) ;
	bool lat_prop_exists      = MAP_GETCASEINSENSITIVE (config, "latitudeProperty",  lat_prop) ;
	bool lon_prop_exists      = MAP_GETCASEINSENSITIVE (config, "longitudeProperty", lon_prop) ;
	bool heur_scale_exists    = MAP_GETCASEINSENSITIVE (config, "heuristicScale",    heur_scale) ;
	bool path_count_exists    = MAP_GETCASEINSENSITIVE (config, "pathCount",         path_count) ;

	if(!start_exists || !end_exists) {
		ErrorCtx_SetError(EMSG_SPPATH_REQUIRED);
		return false;
	}
	if(SI_TYPE(start) != T_NODE || SI_TYPE(end) != T_NODE) {
		ErrorCtx_SetError(EMSG_SPPATH_INVALID_TYPE);
		return false;
	}

	// latitudeProperty/longitudeProperty are what distinguishes this
	// procedure from algo.SPpaths: they're what the A* heuristic is built
	// from, so unlike weightProp/costProp they're mandatory, not optional.
	if(!lat_prop_exists || !lon_prop_exists) {
		ErrorCtx_SetError(EMSG_ASTAR_LATLON_REQUIRED);
		return false;
	}
	if(!(SI_TYPE(lat_prop) & T_STRING) || !(SI_TYPE(lon_prop) & T_STRING)) {
		ErrorCtx_SetError(EMSG_MUST_BE, "latitudeProperty/longitudeProperty", "string");
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

	GraphContext *gc = QueryCtx_GetGraphCtx();
	Graph *g = QueryCtx_GetGraph();
	RelationID *types = NULL;
	uint types_count = 0;
	if(relationships_exists) {
		if(SI_TYPE(relationships) != T_ARRAY ||
			!SIArray_AllOfType(relationships, T_STRING)) {
			ErrorCtx_SetError(EMSG_MUST_BE, "relTypes", "array of strings");
			return false;
		}
		types_count = SIArray_Length(relationships);
		if(types_count > 0) {
			types = arr_new(RelationID, types_count);
			for(uint i = 0; i < types_count; i++) {
				SIValue rel = SIArray_Get(relationships, i);
				const char *type = rel.stringval;
				Schema *s = GraphContext_GetSchema(gc, type, SCHEMA_EDGE);
				if(s == NULL) continue;

				// skip a relation type listed more than once: a duplicate
				// would make the search scan that relation's edges twice,
				// double-counting paths. the small relTypes list makes this
				// linear check cheap.
				RelationID rid = Schema_GetID(s);
				bool dup = false;
				for(uint j = 0; j < arr_len(types); j++) {
					if(types[j] == rid) { dup = true; break; }
				}
				if(!dup) arr_append(types, rid);
			}
			types_count = arr_len(types);
		}
	} else {
		// no relTypes specified: traverse every relation type. expand to
		// concrete relation ids up front (rather than passing the
		// GRAPH_NO_RELATION wildcard through) so each one can be resolved
		// to a matrix and cached once below.
		types_count = Graph_RelationTypeCount(g);
		types = arr_new(RelationID, types_count);
		for(uint i = 0; i < types_count; i++) {
			arr_append(types, (RelationID)i);
		}
	}

	ctx->g             = g;
	ctx->dir           = direction;
	ctx->relationIDs   = types;
	ctx->relationCount = types_count;
	ctx->src           = *(Node *)start.ptrval;
	ctx->dst           = (Node *)end.ptrval;
	ctx->path_count    = 1;

	// resolve and synchronize each relation's matrix once, up front, instead
	// of on every neighbor-expansion call during traversal: the procedure
	// runs under the graph's read lock for its entire lifetime, so the
	// matrices are guaranteed stable for as long as they're cached here
	ctx->relationMatrices = arr_new(Tensor, types_count);
	for(uint i = 0; i < types_count; i++) {
		Tensor R = Graph_GetRelationMatrix(g, types[i], false);
		arr_append(ctx->relationMatrices, R);
	}

	ctx->weight_prop = ATTRIBUTE_ID_NONE;

	if(weight_prop_exists) {
		if(!(SI_TYPE(weight_prop) & T_STRING)) {
			ErrorCtx_SetError(EMSG_MUST_BE, "weightProp", "string");
			return false;
		}
		ctx->weight_prop = GraphContext_GetAttributeID(gc, weight_prop.stringval);
	}

	// heuristicScale converts the haversine heuristic (meters) into weightProp's
	// units. It must be a non-negative lower bound on the weight accrued per
	// meter of straight-line progress for A* to stay admissible (see AStar.h);
	// e.g. 1 when weightProp is a distance in meters, or 1/max_speed when it is
	// travel time. Defaults to 1 (weightProp assumed to be a distance in meters,
	// the historical A* contract).
	ctx->heur_scale = 1.0;

	if(heur_scale_exists) {
		if(!(SI_TYPE(heur_scale) & SI_NUMERIC)) {
			ErrorCtx_SetError(EMSG_MUST_BE, "heuristicScale", "a number");
			return false;
		}
		double s = SI_GET_NUMERIC(heur_scale);
		if(s < 0) {
			ErrorCtx_SetError(EMSG_MUST_BE, "heuristicScale", "a non-negative number");
			return false;
		}
		ctx->heur_scale = s;
	}

	if(path_count_exists) {
		if(SI_TYPE(path_count) != T_INT64) {
			ErrorCtx_SetError(EMSG_MUST_BE, "pathCount", "integer");
			return false;
		}
		// unlike algo.SPpaths, A* is inherently a single-goal directed search;
		// pathCount == 0 (all-minimal) has no meaning here, so require >= 1.
		if(path_count.longval < 1) {
			ErrorCtx_SetError(EMSG_MUST_BE, "pathCount", "a positive integer");
			return false;
		}
		ctx->path_count = SI_GET_NUMERIC(path_count);
	}

	// unlike weightProp, an unresolvable latitudeProperty/
	// longitudeProperty (i.e. no node in the graph ever used that
	// property key) isn't treated as an error: A* already degrades
	// gracefully to h == 0 (i.e. plain Dijkstra) whenever a node's
	// coordinates can't be read, so ATTRIBUTE_ID_NONE here just means that
	// degradation applies graph-wide.
	ctx->lat_prop = GraphContext_GetAttributeID(gc, lat_prop.stringval);
	ctx->lon_prop = GraphContext_GetAttributeID(gc, lon_prop.stringval);

	return true;
}

static ProcedureResult Proc_AStarPathsInvoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {
	AStarCtx *actx = rm_calloc(1, sizeof(AStarCtx));
	if(!validate_config(args[0], actx)) {
		AStarCtx_Free(actx);
		return PROCEDURE_ERR;
	}

	ctx->privateData = actx;
	_process_yield(actx, yield);

	// src == dst is degenerate: a path needs at least one edge. rather than
	// have the search trivially "find" src at distance 0 with zero edges
	// traversed, just report no results, mirroring algo.SPpaths' handling.
	bool src_eq_dst =
		(ENTITY_GET_ID(&actx->src) == ENTITY_GET_ID(actx->dst));

	if(src_eq_dst) {
		actx->paths   = arr_new(Path *, 0);
		actx->weights = arr_new(double, 0);
		return PROCEDURE_OK;
	}

	NodeID src_id = ENTITY_GET_ID(&actx->src);
	NodeID dst_id = ENTITY_GET_ID(actx->dst);

	if(actx->path_count == 1) {
		// single shortest path: plain A*.
		Path   *path;
		double  weight;
		bool found = AStar_ShortestPath(&path, &weight, actx->g, src_id, dst_id,
				actx->dir, actx->relationIDs, actx->relationMatrices,
				actx->relationCount, actx->weight_prop, actx->lat_prop,
				actx->lon_prop, actx->heur_scale);

		actx->paths   = arr_new(Path *, found ? 1 : 0);
		actx->weights = arr_new(double, found ? 1 : 0);
		if(found) {
			arr_append(actx->paths, path);
			arr_append(actx->weights, weight);
		}
	} else {
		// k shortest loopless paths: Yen driven by A* spur searches.
		AStar_KShortestPaths(actx->g, src_id, dst_id, actx->path_count,
				actx->dir, actx->relationIDs, actx->relationMatrices,
				actx->relationCount, actx->weight_prop, actx->lat_prop,
				actx->lon_prop, actx->heur_scale, &actx->paths, &actx->weights);
	}

	return PROCEDURE_OK;
}

static SIValue *Proc_AStarPathsStep
(
	ProcedureCtx *ctx
) {
	ASSERT(ctx->privateData != NULL);

	AStarCtx *actx = ctx->privateData;

	if(actx->paths == NULL || actx->cursor >= arr_len(actx->paths)) {
		return NULL;
	}

	// hand off the next result path; advance the cursor so AStarCtx_Free
	// (called later) never re-frees a path SIPath_Wrap/Path_Free consumed here.
	Path  *path   = actx->paths[actx->cursor];
	double weight = actx->weights[actx->cursor];
	actx->cursor++;

	if(actx->yield_path) {
		*actx->yield_path = SIPath_Wrap(&path);
	} else {
		Path_Free(path);
	}

	if(actx->yield_path_weight) *actx->yield_path_weight = SI_DoubleVal(weight);

	return actx->output;
}

static ProcedureResult Proc_AStarPathsFree
(
	ProcedureCtx *ctx
) {
	ASSERT(ctx != NULL);

	AStarCtx *actx = ctx->privateData;
	AStarCtx_Free(actx);

	return PROCEDURE_OK;
}

ProcedureCtx *Proc_AStarPathCtx(void) {
	ProcedureOutput output;
	void *privateData = NULL;

	ProcedureOutput *outputs = arr_newlen(ProcedureOutput, 2);

	outputs[0] = (ProcedureOutput){.name = "path",       .type = T_PATH};
	outputs[1] = (ProcedureOutput){.name = "pathWeight", .type = T_DOUBLE};

	return ProcCtxNew("algo.AStar", 1, outputs, Proc_AStarPathsStep,
			Proc_AStarPathsInvoke, Proc_AStarPathsFree, privateData, true);
}
