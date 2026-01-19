/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"

// initialize traversal configuration
bool traverse_init_config
(
	JSContext *js_ctx,          // javascript context
	int argc,                   // # arguments (expecting 0 or 1)
	JSValueConst *argv,         // config arguments (expecting a map)

	uint *distance,             // [output] traverse depth distance
	char ***labels,             // [output] restrict to reachable nodes of lbl
	char ***rel_types,          // [output] restrict to specified rel-types
	GRAPH_EDGE_DIR *dir,        // [output] traverse direction
	GraphEntityType *ret_type,  // [output] return either nodes or edges
	const char **err_msg        // [output] report error message
);

// traverse from src node
// caller can specify the set of relationship-types to consider
// in addition to the labels associated with reachable nodes
// this function can traverse in both directions (forward, backwards or both)
// lastly caller can decide if he wish to get the reachable nodes or edges
GraphEntity **traverse
(
	const EntityID *sources,  // traversal begins here
	uint n,                   // number of sources
	uint distance,            // direct neighbors [ignored]
	const char **labels,      // neighbors labels
	const char **rel_types,   // edge types to consider
	GRAPH_EDGE_DIR dir,       // edge direction
	GraphEntityType ret_type  // returned entity type
);

