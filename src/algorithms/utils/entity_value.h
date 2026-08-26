/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../value.h"
#include "../../graph/entities/graph_entity.h"

// read a numeric attribute from a graph entity, returning 'default_value' when
// the attribute is absent or not numeric. shared by the shortest-path engines
// (Dijkstra, A*, Yen, all-shortest DAG) which all read a per-edge weight/cost
// this way.
static inline SIValue _get_value_or_default
(
	GraphEntity *ge,
	AttributeID id,
	SIValue default_value
) {
	SIValue v;

	if(!GraphEntity_GetProperty(ge, id, &v)) {
		return default_value;
	}

	if(SI_TYPE(v) & SI_NUMERIC) {
		return v;
	}

	return default_value;
}
