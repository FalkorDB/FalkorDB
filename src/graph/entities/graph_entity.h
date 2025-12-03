/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "attribute_set.h"
#include "../../value.h"
#include "../../../deps/GraphBLAS/Include/GraphBLAS.h"

#define ENTITY_ID_ISLT(a, b) ((*a) < (*b))
#define INVALID_ENTITY_ID -1l

#define ENTITY_GET_ID(graphEntity) (graphEntity)->id

// Defined in graph_entity.c
extern SIValue *ATTRIBUTE_NOTFOUND;

typedef GrB_Index EdgeID;
typedef GrB_Index NodeID;
typedef GrB_Index EntityID;

/*  Format a graph entity string according to the enum.
    One can sum the enum values in order to print multiple value:
    ENTITY_ID + ENTITY_LABELS_OR_RELATIONS will print both id and label. */

typedef enum {
	ENTITY_ID = 1,                       // print id only
	ENTITY_LABELS_OR_RELATIONS = 1 << 1, // print label or relationship type
	ENTITY_PROPERTIES = 1 << 2           // print properties
} GraphEntityStringFormat;

typedef enum GraphEntityType {
	GETYPE_UNKNOWN,
	GETYPE_NODE,
	GETYPE_EDGE
} GraphEntityType;

// Common denominator between nodes and edges.
typedef struct {
	AttributeSet *attributes;
	EntityID id;
} GraphEntity;

// adds property to entity
// returns - reference to newly added property
bool GraphEntity_AddProperty
(
	GraphEntity *e,
	AttributeID attr_id,
	SIValue value
);

// returns true if entity contains attribute
bool GraphEntity_ContainsProperty
(
	const GraphEntity *e,  // graph entity
	AttributeID attr_id
);

// retrieves entity's property
// if the key does not exist returns false
bool GraphEntity_GetProperty
(
	const GraphEntity *e,  // graph entity
	AttributeID attr_id,   // attribute id
	SIValue *v             // attribute value
);

// returns an SIArray of all keys in graph entity properties
SIValue GraphEntity_Keys
(
	const GraphEntity *e
);

// returns an SIArray of all keys and values in graph entity properties.
// Keys at even positions, Values at odd position of the array
SIValue GraphEntity_Properties
(
	const GraphEntity *e
);

// prints the graph entity into a buffer, returns what is the string length
// buffer can be re-allocated if needed
void GraphEntity_ToString
(
	const GraphEntity *e,
	char **buffer,
	size_t *bufferLen,
	size_t *bytesWritten,
	GraphEntityStringFormat format,
	GraphEntityType entityType
);

// returns true if the given graph entity has been deleted
bool GraphEntity_IsDeleted
(
	const GraphEntity *e
);

// returns attribute-set of entity
AttributeSet GraphEntity_GetAttributes
(
	const GraphEntity *e
);

int GraphEntity_ClearAttributes
(
	GraphEntity *e
);
