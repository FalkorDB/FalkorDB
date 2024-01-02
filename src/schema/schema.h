/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../redismodule.h"
#include "../index/index.h"
#include "redisearch_api.h"
#include "../constraint/constraint.h"
#include "../graph/entities/graph_entity.h"
#include "../graph/entities/attribute_set.h"

#define ACTIVE_IDX(s)  s->index[0]
#define PENDING_IDX(s) s->index[1]

typedef enum SchemaType {
	SCHEMA_NODE,
	SCHEMA_EDGE,
} SchemaType;

// schema represents the structure of a typed graph entity (Node/Edge)
// similar to a relational table structure, our schemas are a collection
// of attributes we've encountered overtime as entities were created or updated
typedef struct {
	int id;                   // schema id
	char *name;               // schema name
	SchemaType type;          // schema type (node/edge)
	Index index[2];           // active/pending index
	Constraint *constraints;  // constraints array
} Schema;

// creates a new schema
Schema *Schema_New
(
	SchemaType type,
	int id,
	const char *name
);

// return the given schema's name
const char *Schema_GetName
(
	const Schema *s
);

// return the given schema's ID
int Schema_GetID
(
	const Schema *s
);

const char *Schema_GetName
(
	const Schema *s
);

// return schema type
SchemaType Schema_GetType
(
	const Schema *s
);

// returns true if schema has an index
bool Schema_HasIndices
(
	const Schema *s
);

// retrieves all indicies from schema
// active index
// pending index
// returns number of indicies set
unsigned short Schema_GetIndicies
(
	const Schema *s,
	Index indicies[2]
);

// get index from schema
// returns NULL if index wasn't found
Index Schema_GetIndex
(
	const Schema *s,           // schema to get index from
	const AttributeID *attrs,  // indexed attributes
	uint n,                    // number of attributes
	IndexFieldType t,          // all index attributes must be of this type
	bool include_pending       // take into considiration pending indicies
);

// assign a new attribute to index
// attribute must not be associated with an index
int Schema_AddIndex
(
	Index *idx,         // [input/output] index to create
	Schema *s,          // schema holding the index
	IndexField *fields  // field to index
);

// removes index
int Schema_RemoveIndex
(
	Schema *s,        // schema to remove index from
	const char *f,    // field to remove from index
	IndexFieldType t  // field type
);

// activate pending index
// asserts that pending index is enabled
// drops current active index if exists
void Schema_ActivateIndex
(
	Schema *s   // schema to activate index on
);

// introduce node to schema index
void Schema_AddNodeToIndex
(
	const Schema *s,
	const Node *n
);

// introduce edge to schema index
void Schema_AddEdgeToIndex
(
	const Schema *s,
	const Edge *e
);

// remove node from schema index
void Schema_RemoveNodeFromIndex
(
	const Schema *s,
	const Node *n
);

// remove edge from schema index
void Schema_RemoveEdgeFromIndex
(
	const Schema *s,
	const Edge *e
);

// Free schema
void Schema_Free
(
	Schema *s
);

//------------------------------------------------------------------------------
// constraints API
//------------------------------------------------------------------------------

// check if schema has constraints
bool Schema_HasConstraints
(
	const Schema *s  // schema to query
);

// checks if schema constains constraint
bool Schema_ContainsConstraint
(
	const Schema *s,           // schema to search
	ConstraintType t,          // constraint type
	const AttributeID *attrs,  // constraint attributes
	uint attr_count            // number of attributes
);

// retrieves constraint 
// returns NULL if constraint was not found
Constraint Schema_GetConstraint
(
	const Schema *s,           // schema from which to get constraint
	ConstraintType t,          // constraint type
	const AttributeID *attrs,  // constraint attributes
	uint attr_count            // number of attributes
);

// get all constraints in schema
const Constraint *Schema_GetConstraints
(
	const Schema *s  // schema from which to extract constraints
);

// adds a constraint to schema
void Schema_AddConstraint
(
	Schema *s,       // schema holding the index
	Constraint c     // constraint to add
);

// removes constraint from schema
void Schema_RemoveConstraint
(
	Schema *s,    // schema
	Constraint c  // constraint to remove
);

// enforce all constraints under given schema on entity
bool Schema_EnforceConstraints
(
	const Schema *s,       // schema
	const GraphEntity *e,  // entity to enforce
	char **err_msg         // report error message
);

