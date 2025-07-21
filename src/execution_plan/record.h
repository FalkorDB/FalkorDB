/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "../value.h"
#include "../../deps/rax/rax.h"
#include "../graph/entities/node.h"
#include "../graph/entities/edge.h"
#include <sys/types.h>

// Return value in case of call to Record_GetEntryIdx with invalid entry alias.
#define INVALID_INDEX -1

typedef enum  {
	REC_TYPE_UNKNOWN = 0,
	REC_TYPE_SCALAR = 1 << 0,
	REC_TYPE_NODE = 1 << 1,
	REC_TYPE_EDGE = 1 << 2,
	REC_TYPE_HEADER = 1 << 3,
} RecordEntryType;

typedef struct {
	union {
		SIValue s;
		Node n;
		Edge e;
	} value;
	RecordEntryType type;
} Entry;

typedef struct {
	void *owner;        // Owner of record.
	rax *mapping;       // Mapping between alias to record entry.
	Entry entries[];    // Array of entries.
} _Record;

typedef _Record *Record;

// create a new record sized to accommodate all entries in the given map
Record Record_New
(
	rax *mapping
);

// clones record
void Record_Clone
(
	const Record r,
	Record clone
);

// deep clone record
void Record_DeepClone
(
	const Record r,
	Record clone
);

// merge record b into a, sharing any nested references in b with a
void Record_Merge
(
	Record a,
	const Record b
);

// merge entries from `from` into `to`, transfer ownership if transfer_ownership
// is on
void Record_TransferEntries
(
	Record *to,              // destination record
	Record from,             // src record
	bool transfer_ownership  // transfer ownership of the record to dest or not
);

// returns number of entries record can hold
uint Record_length
(
	const Record r
);

// return true if records contains entry at position 'idx'
bool Record_ContainsEntry
(
	const Record r,
	uint idx
);

// return alias position within the record
uint Record_GetEntryIdx
(
	Record r,
	const char *alias
);

// get entry type
RecordEntryType Record_GetType
(
	const Record r,
	uint idx
);

// get a node from record at position idx
Node *Record_GetNode
(
	const Record r,
	uint idx
);

// get an edge from record at position idx
Edge *Record_GetEdge
(
	const Record r,
	uint idx
);

// get an SIValue containing the entity at position idx
SIValue Record_Get
(
	Record r,
	uint idx
);

// remove item at position idx
void Record_Remove
(
	Record r,
	uint idx
);

// get a graph entity from record at position idx
GraphEntity *Record_GetGraphEntity
(
	const Record r,
	uint idx
);

// add a scalar, node, or edge to the record, depending on the SIValue type
void Record_Add
(
	Record r,
	uint idx,
	SIValue v
);

// add a scalar to record at position idx and return a reference to it
SIValue *Record_AddScalar
(
	Record r,
	uint idx,
	SIValue v
);

// add a node to record at position idx and return a reference to it
Node *Record_AddNode
(
	Record r,
	uint idx,
	Node node
);

// add an edge to record at position idx and return a reference to it
Edge *Record_AddEdge
(
	Record r,
	uint idx,
	Edge edge
);

// ensure that all scalar values in record are access-safe
void Record_PersistScalars
(
	Record r
);

// string representation of record
size_t Record_ToString
(
	const Record r,
	char **buf,
	size_t *buf_cap
);

// retrieves mapping associated with record
rax *Record_GetMappings
(
	const Record r
);

// remove and free entry at position idx
void Record_FreeEntry
(
	Record r,
	int idx
);

// free record entries
void Record_FreeEntries
(
	Record r
);

// Free record.
void Record_Free
(
	Record r
);

