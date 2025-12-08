/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "index_field.h"
#include "redisearch_api.h"
#include "../graph/graph.h"
#include "../graph/entities/node.h"
#include "../graph/entities/edge.h"
#include "../filter_tree/filter_tree.h"
#include "../graph/entities/graph_entity.h"

#define INDEX_OK   1  // index operation succeeded
#define INDEX_FAIL 0  // index operation failed
#define INDEX_SEPARATOR '\1'  // can't use '\0', RediSearch will terminate on \0

// forward declaration
typedef struct _Index _Index;
typedef _Index *Index;

// deprecated
typedef enum {
	IDX_ANY         = 0,
	IDX_EXACT_MATCH = 1,
	IDX_FULLTEXT    = 2,
} IndexType;

// edge document key
typedef struct {
	EntityID src_id;   // edge source node ID
	EntityID dest_id;  // edge destination node ID
	EntityID edge_id;  // edge ID
} EdgeIndexKey;

// create a new index
Index Index_New
(
	const char *label,           // indexed label
	int label_id,                // indexed label id
	GraphEntityType entity_type  // entity type been indexed
);

// clone index
Index Index_Clone
(
	const Index idx  // index to clone
);

// create range index
Index Index_RangeCreate
(
	const char *label,            // label/relationship type
	GraphEntityType entity_type,  // entity type (node/edge)
	const char *attr,             // attribute to index
	AttributeID attr_id           // attribute id
);

// create fulltext index
Index Index_FulltextCreate
(
	const char *label,            // label/relationship type
	GraphEntityType entity_type,  // entity type (node/edge)
	const char *attribute,        // attribute to index
	AttributeID attr_id,          // attribute id
	const SIValue options         // index options
);

// create a vector index
Index Index_VectorCreate
(
	const char *label,            // label/relationship type
	GraphEntityType entity_type,  // entity type (node/edge)
	const char *attr,             // attribute to index
	AttributeID attr_id,          // attribute id
	SIValue options               // index options
);

// returns number of pending changes
int Index_PendingChanges
(
	const Index idx  // index to inquery
);

// try to enable index by dropping number of pending changes by 1
// the index is enabled once there are no pending changes
void Index_Enable
(
	Index idx // index to enable
);

// disable index by increasing the number of pending changes
// and re-creating the internal RediSearch index
void Index_Disable
(
	Index idx  // index to disable
);

// returns true if index doesn't contains any pending changes
bool Index_Enabled
(
	const Index idx  // index to get state of
);

// returns RediSearch index
RSIndex *Index_RSIndex
(
	const Index idx  // index to get internal RediSearch index from
);

// responsible for creating the index structure only!
// e.g. fields, stopwords, language
void Index_ConstructStructure
(
	Index idx
);

// populates index
void Index_Populate
(
	Index idx,  // index to populate
	Graph *g    // graph holding entities to index
);

// adds field to index
int Index_AddField
(
	Index idx,         // index modified
	IndexField *field  // field added
);

// removes field from index
void Index_RemoveField
(
	Index idx,             // index modified
	AttributeID attr_id,   // field to remove
	IndexFieldType t       // field type
);

// index node
void Index_IndexNode
(
	Index idx,     // index to populate
	const Node *n  // node to index
);

// index edge
void Index_IndexEdge
(
	Index idx,     // index to populate
	const Edge *e  // edge to index
);

// remove node from index
void Index_RemoveNode
(
	Index idx,     // index to update
	const Node *n  // node to remove from index
);

// remove edge from index
void Index_RemoveEdge
(
	Index idx,     // index to update
	const Edge *e  // edge to remove from index
);

//------------------------------------------------------------------------------
// index query API
//------------------------------------------------------------------------------

// construct a query tree from a filter tree
RSQNode *Index_BuildQueryTree
(
	FT_FilterNode **none_converted_filters,  // [out] none converted filters
	const Index idx,                         // index to query
	const FT_FilterNode *tree                // filter tree to convert
);

// construct a vector query tree
RSQNode *Index_BuildVectorQueryTree
(
	const Index idx,    // index to query
	const char *field,  // field to query
	const float *vec,   // query vector
	size_t nbytes,      // vector size in bytes
	int k			    // number of results to return
);

// construct a unique constraint query tree
RSQNode *Index_BuildUniqueConstraintQuery
(
	const Index idx,           // index to query
	const SIValue *attr_vals,  // entity attributes to query
	AttributeID *attr_ids,     // constraint attribute ids
	uint8_t n                  // number of constraint attributes
);

// query an index
RSResultsIterator *Index_Query
(
	const Index idx,    // index to query
	const char *query,  // query to execute
	char **err          // [optional] report back error
);

// returns index graph entity type
GraphEntityType Index_GraphEntityType
(
	const Index idx
);

// returns number of fields indexed
uint Index_FieldsCount
(
	const Index idx  // index to query
);

// returns a shallow copy of indexed fields
const IndexField *Index_GetFields
(
	const Index idx  // index to query
);

// retrieve field by id
// returns NULL if field does not exist
IndexField *Index_GetField
(
	int *pos,         // [optional out] field index
	const Index idx,  // index to get field from
	AttributeID id    // field attribute id
);

// returns indexed field type
// if field is not indexed, INDEX_FLD_UNKNOWN is returned
IndexFieldType Index_GetFieldType
(
	const Index idx,     // index to query
	AttributeID attr_id  // field to retrieve type of
);

// checks if index contains field
// returns true if field is indexed, false otherwise
bool Index_ContainsField
(
	const Index idx,     // index to query
	AttributeID id,      // field to look for
	IndexFieldType type  // field type to look for
);

// returns indexed label
const char *Index_GetLabel
(
	const Index idx  // index to query
);

// returns indexed label ID
int Index_GetLabelID
(
	const Index idx  // index to query
);

// returns indexed language
const char *Index_GetLanguage
(
	const Index idx  // index to query
);

// check if index contains stopwords
bool Index_ContainsStopwords
(
	const Index idx  // index to query
);

// returns indexed stopwords
char **Index_GetStopwords
(
	const Index idx,   // index to query
	size_t *size       // number of stopwords
);

// set indexed language
bool Index_SetLanguage
(
	Index idx,            // index modified
	const char *language  // language to set
);

// set indexed stopwords
bool Index_SetStopwords
(
	Index idx,         // index modified
	char ***stopwords  // stopwords
);

// free index
void Index_Free
(
	Index idx  // index being freed
);

