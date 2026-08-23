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
//                   longitudeProperty: 'lon'}) YIELD path, pathWeight
// RETURN path, pathWeight

typedef struct {
	Node src;                    // source node
	Node *dst;                   // destination node, non-owning: points into
	                              // the caller-owned SIValue passed as
	                              // 'targetNode', same convention as
	                              // proc_sp_paths.c's SinglePairCtx.dst
	Graph *g;                    // graph to traverse
	int *relationIDs;            // edge type(s) to traverse
	Tensor *relationMatrices;    // relation matrix per relationIDs entry
	int relationCount;           // length of relationIDs
	GRAPH_EDGE_DIR dir;          // traverse direction
	AttributeID weight_prop;     // weight attribute id
	AttributeID lat_prop;        // latitude attribute id (required)
	AttributeID lon_prop;        // longitude attribute id (required)
	Path *path;                  // result path, NULL if not found / already yielded
	double weight;               // result path weight
	bool done;                   // true once Step has emitted the single result
	SIValue output[2];           // result returned
	SIValue *yield_path;         // yield path
	SIValue *yield_path_weight;  // yield path weight
} AStarCtx;

// free AStarCtx
static void AStarCtx_Free
(
	AStarCtx *ctx
) {
	if(ctx == NULL) return;

	if(ctx->path)             Path_Free(ctx->path);
	if(ctx->relationIDs)      arr_free(ctx->relationIDs);
	if(ctx->relationMatrices) arr_free(ctx->relationMatrices);

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

// validate config map and initialize AStarCtx
static ProcedureResult validate_config
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

	bool start_exists         = MAP_GET(config, "sourceNode",        start);
	bool end_exists           = MAP_GET(config, "targetNode",        end);
	bool relationships_exists = MAP_GET(config, "relTypes",          relationships);
	bool dir_exists           = MAP_GET(config, "relDirection",      dir);
	bool weight_prop_exists   = MAP_GET(config, "weightProp",        weight_prop);
	bool lat_prop_exists      = MAP_GET(config, "latitudeProperty",  lat_prop);
	bool lon_prop_exists      = MAP_GET(config, "longitudeProperty", lon_prop);

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
			for(uint i = 0; i < types_count; i++) {
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
		// to a matrix and cached once below.
		types_count = Graph_RelationTypeCount(g);
		types = arr_new(int, types_count);
		for(uint i = 0; i < types_count; i++) {
			arr_append(types, (int)i);
		}
	}

	ctx->g             = g;
	ctx->dir           = direction;
	ctx->relationIDs   = types;
	ctx->relationCount = types_count;
	ctx->src           = *(Node *)start.ptrval;
	ctx->dst           = (Node *)end.ptrval;

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

	// unlike weightProp, an unresolvable latitudeProperty/
	// longitudeProperty (i.e. no node in the graph ever used that
	// property key) isn't treated as an error: AStar_ShortestPath already
	// degrades gracefully to h == 0 (i.e. plain Dijkstra) whenever a
	// node's coordinates can't be read, so ATTRIBUTE_ID_NONE here just
	// means that degradation applies graph-wide.
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

	// src == dst is degenerate: a path needs at least one edge. rather
	// than have the search trivially "find" src at distance 0 with zero
	// edges traversed, just report not-found, mirroring algo.SPpaths'
	// identical src == dst handling.
	bool src_eq_dst =
		(ENTITY_GET_ID(&actx->src) == ENTITY_GET_ID(actx->dst));

	if(src_eq_dst) {
		return PROCEDURE_OK;
	}

	Path *path;
	double weight;

	bool found = AStar_ShortestPath(&path, &weight, actx->g,
			ENTITY_GET_ID(&actx->src), ENTITY_GET_ID(actx->dst), actx->dir,
			actx->relationIDs, actx->relationMatrices, actx->relationCount,
			actx->weight_prop, actx->lat_prop, actx->lon_prop);

	if(found) {
		actx->path   = path;
		actx->weight = weight;
	}

	return PROCEDURE_OK;
}

static SIValue *Proc_AStarPathsStep
(
	ProcedureCtx *ctx
) {
	ASSERT(ctx->privateData != NULL);

	AStarCtx *actx = ctx->privateData;

	if(actx->done || actx->path == NULL) {
		return NULL;
	}

	// mark done and detach the path from the context before handing it
	// off below, so AStarCtx_Free (called later, regardless of whether
	// SIPath_Wrap or Path_Free consumed it here) never double-frees it.
	actx->done = true;
	Path *path = actx->path;
	actx->path = NULL;

	if(actx->yield_path) {
		*actx->yield_path = SIPath_Wrap(&path);
	} else {
		Path_Free(path);
	}

	if(actx->yield_path_weight) *actx->yield_path_weight = SI_DoubleVal(actx->weight);

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
