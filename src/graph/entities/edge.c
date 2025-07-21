/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include <stdio.h>
#include <stdlib.h>

#include "RG.h"
#include "edge.h"
#include "../graph.h"
#include "graph_entity.h"

NodeID Edge_GetSrcNodeID
(
	const Edge *edge
) {
	ASSERT(edge);
	return edge->src_id;
}

NodeID Edge_GetDestNodeID
(
	const Edge *edge
) {
	ASSERT(edge);
	return edge->dest_id;
}

int Edge_GetRelationID
(
	const Edge *edge
) {
	ASSERT(edge);
	ASSERT(edge->relationID != GRAPH_NO_RELATION);
	ASSERT(edge->relationID != GRAPH_UNKNOWN_RELATION);
	return edge->relationID;
}

void Edge_SetSrcNodeID
(
	Edge *e,
	NodeID id
) {
	ASSERT(e);
	e->src_id = id;
}

void Edge_SetDestNodeID
(
	Edge *e,
	NodeID id
) {
	ASSERT(e);
	e->dest_id = id;
}

void Edge_SetRelationID
(
	Edge *e,
	RelationID relationID
) {
	ASSERT(e);
	e->relationID = relationID;
}

void Edge_ToString
(
	const Edge *e,
	char **buffer,
	size_t *bufferLen,
	size_t *bytesWritten,
	GraphEntityStringFormat format
) {
	GraphEntity_ToString((const GraphEntity *)e, buffer, bufferLen, bytesWritten, format, GETYPE_EDGE);
}

void Edge_Free
(
	Edge *edge
) {
	if(!edge) return;

	free(edge);
}

