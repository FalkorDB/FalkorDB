/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "schema.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../errors/errors.h"
#include "../index/indexer.h"
#include "../graph/graphcontext.h"
#include "../constraint/constraint.h"

// add an exact match index to schema
static int Schema_AddExactMatchIndex
(
	Index *idx,        // [input/output] index to create
	Schema *s,         // schema holding the index
	IndexField *field  // field to index
) {
	ASSERT(s != NULL);
	ASSERT(idx != NULL);
	ASSERT(field != NULL);

	// see if index already exists
	Index _idx = NULL;
	GraphEntityType et = (s->type == SCHEMA_NODE) ? GETYPE_NODE : GETYPE_EDGE;

	//--------------------------------------------------------------------------
	// determine which index to populate
	//--------------------------------------------------------------------------

	// if pending-index exists, reuse it
	// if active-index exists, clone it and use clone
	// otherwise (first index) create a new index
	Index active  = ACTIVE_EXACTMATCH_IDX(s);
	Index pending = PENDING_EXACTMATCH_IDX(s);
	Index altered = (pending != NULL) ? pending : active;

	if(altered != NULL) {
		// make sure attribute isn't already indexed
		if(Index_ContainsAttribute(altered, field->id)) {
			// field already indexed, quick return
			IndexField_Free(field);
			return INDEX_FAIL;
		}
	}

	_idx = pending;
	if(pending == NULL) {
		if(active != NULL) {
			_idx = Index_Clone(active);
		} else {
			_idx = Index_New(s->name, s->id, IDX_EXACT_MATCH, et);
		}
	}
	PENDING_EXACTMATCH_IDX(s) = _idx;  // set pending exact-match index

	Index_AddField(_idx, field);

	*idx = _idx;
	return INDEX_OK;
}

// add a full-text index to schema
static int Schema_AddFullTextIndex
(
	Index *idx,        // [input/output] index to create
	Schema *s,         // schema holding the index
	IndexField *field  // field to index
) {
	ASSERT(s != NULL);
	ASSERT(idx != NULL);
	ASSERT(field != NULL);

	// see if index already exists
	Index _idx = NULL;

	//--------------------------------------------------------------------------
	// determine which index to populate
	//--------------------------------------------------------------------------

	// if pending-index exists, reuse it
	// if active-index exists, clone it and use clone
	// otherwise (first index) create a new index
	Index active  = ACTIVE_FULLTEXT_IDX(s);
	Index pending = PENDING_FULLTEXT_IDX(s);
	Index altered = (pending != NULL) ? pending : active;

	if(altered != NULL) {
		// make sure attribute isn't already indexed
		if(Index_ContainsAttribute(altered, field->id)) {
			// field already indexed, quick return
			IndexField_Free(field);
			return INDEX_FAIL;
		}
	}

	_idx = pending;
	if(pending == NULL) {
		if(active != NULL) {
			_idx = Index_Clone(active);
		} else {
			_idx = Index_New(s->name, s->id, IDX_FULLTEXT, GETYPE_NODE);
		}
	}
	PENDING_FULLTEXT_IDX(s) = _idx;  // set pending full-text index

	Index_AddField(_idx, field);

	*idx = _idx;
	return INDEX_OK;
}

static int _Schema_RemoveExactMatchIndex
(
	Schema *s,
	const char *field
) {
	ASSERT(s != NULL);
	ASSERT(field != NULL);

	GraphContext *gc = QueryCtx_GetGraphCtx();

	// convert attribute name to attribute ID
	Attribute_ID attr_id = GraphContext_GetAttributeID(gc, field);
	if(attr_id == ATTRIBUTE_ID_NONE) {
		return INDEX_FAIL;
	}

	// try to get index
	// if a pending index exists use it otherwise use the active index
	Index active  = ACTIVE_EXACTMATCH_IDX(s);
	Index pending = PENDING_EXACTMATCH_IDX(s);
	Index idx     = pending;

	// both pending and active indicies do not exists
	if(pending == NULL) {
		if(active == NULL) {
			return INDEX_FAIL;
		}
		// use active
		pending = Index_Clone(active);
		PENDING_EXACTMATCH_IDX(s) = pending;
		idx = pending;
	}

	// index doesn't containts attribute
	if(Index_ContainsAttribute(idx, attr_id) == false) {
		return INDEX_FAIL;
	}

	//--------------------------------------------------------------------------
	// make sure index doesn't supports any constraints
	//--------------------------------------------------------------------------

	uint n = array_len(s->constraints);
	for(uint i = 0; i < n; i++) {
		Constraint c = s->constraints[i];
		if(Constraint_GetStatus(c) != CT_FAILED &&
		   Constraint_GetType(c) == CT_UNIQUE   &&
		   Constraint_ContainsAttribute(c, attr_id)) {
			ErrorCtx_SetError(EMSG_INDEX_SUPPORT_CONSTRAINTS);
			return INDEX_FAIL;
		}
	}

	Index_RemoveField(idx, attr_id);

	// if index field count dropped to 0 remove index from schema
	// index will be freed by the indexer thread
	if(Index_FieldsCount(idx) == 0) {
		if(active != NULL) {
			ACTIVE_EXACTMATCH_IDX(s) = NULL;  // disconnect index from schema
			Indexer_DropIndex(active, gc);
		}

		if(pending != NULL) {
			PENDING_EXACTMATCH_IDX(s) = NULL;  // disconnect index from schema
			Indexer_DropIndex(pending, gc);
		}
	} else {
		Indexer_PopulateIndex(gc, s, idx);
	}

	return INDEX_OK;
}

static int _Schema_RemoveFullTextIndex
(
	Schema *s
) {
	// removing a fulltext index is performed in one go
	// the entire index is dropped, even if it contains multiple fields
	// unlike an exact-match index where individual fields can be removed
	ASSERT(s != NULL);

	Index idx     = NULL;
	Index active  = ACTIVE_FULLTEXT_IDX(s);
	Index pending = PENDING_FULLTEXT_IDX(s);

	// both active and pending do not exists, nothing to drop
	if(pending == NULL && active == NULL) {
		return INDEX_FAIL;
	}

	// disconnect both active and pending indicies from schema
	ACTIVE_FULLTEXT_IDX(s)  = NULL;
	PENDING_FULLTEXT_IDX(s) = NULL;

	GraphContext *gc = QueryCtx_GetGraphCtx();

	//--------------------------------------------------------------------------
	// disable and async drop
	//--------------------------------------------------------------------------

	if(active != NULL) {
		Index_Disable(active);
		Indexer_DropIndex(active, gc);
	}

	if(pending != NULL) {
		Index_Disable(pending);
		Indexer_DropIndex(pending, gc);
	}

	return INDEX_OK;
}

static void Schema_ActivateExactMatchIndex
(
	Schema *s   // schema to activate index on
) {
	Index active  = ACTIVE_EXACTMATCH_IDX(s);
	Index pending = PENDING_EXACTMATCH_IDX(s);

	ASSERT(Index_Enabled(pending) == true);

	// drop active if exists
	if(active != NULL) {
		Index_Free(active);
	}

	// set pending index as active
	ACTIVE_EXACTMATCH_IDX(s) = pending;

	// clear pending index
	PENDING_EXACTMATCH_IDX(s) = NULL;

	//--------------------------------------------------------------------------
	// update unique constraint private data
	//--------------------------------------------------------------------------

	active = ACTIVE_EXACTMATCH_IDX(s);

	// uniqie constraint rely on exact-match indicies
	// whenever such an index is updated
	// we need to update the relevant uniqie constraint
	uint n = array_len(s->constraints);
	for(uint i = 0; i < n; i++) {
		Constraint c = s->constraints[i];
		Constraint_SetPrivateData(c, active);
	}
}

static void Schema_ActivateFullTextIdx
(
	Schema *s   // schema to activate index on
) {
	Index active  = ACTIVE_FULLTEXT_IDX(s);
	Index pending = PENDING_FULLTEXT_IDX(s);

	// drop active if exists
	if(active != NULL) {
		Index_Free(active);
	}

	// set pending index as active
	ACTIVE_FULLTEXT_IDX(s) = pending;

	// clear pending index
	PENDING_FULLTEXT_IDX(s) = NULL;
}

Schema *Schema_New
(
	SchemaType type,
	int id,
	const char *name
) {
	ASSERT(name != NULL);

	Schema *s = rm_calloc(1, sizeof(Schema));

	s->id          = id;
	s->type        = type;
	s->name        = rm_strdup(name);
	s->constraints = array_new(Constraint, 0);

	return s;
}

int Schema_ID
(
	const Schema *s
) {
	ASSERT(s != NULL);

	return s->id;
}

const char *Schema_GetName
(
	const Schema *s
) {
	ASSERT(s);
	return s->name;
}

int Schema_GetID
(
	const Schema *s
) {
  ASSERT(s);
  return s->id;
}

// return schema type
SchemaType Schema_GetType
(
	const Schema *s
) {
	ASSERT(s != NULL);
	return s->type;
}

bool Schema_HasIndices
(
	const Schema *s
) {
	ASSERT(s != NULL);

	return (ACTIVE_FULLTEXT_IDX(s)   ||
			PENDING_FULLTEXT_IDX(s)  ||
			ACTIVE_EXACTMATCH_IDX(s) ||
			PENDING_EXACTMATCH_IDX(s));
}

unsigned short Schema_IndexCount
(
	const Schema *s
) {
	ASSERT(s != NULL);
	unsigned short n = 0;

	if(ACTIVE_FULLTEXT_IDX(s) || PENDING_FULLTEXT_IDX(s)) n += 1;
	if(ACTIVE_EXACTMATCH_IDX(s) || PENDING_EXACTMATCH_IDX(s)) n += 1;

	return n;
}

// retrieves all indicies from schema
// active exact-match index
// pending exact-match index
// active fulltext index
// pending fulltext index
// returns number of indicies set
unsigned short Schema_GetIndicies
(
	const Schema *s,
	Index indicies[4]
) {
	int i = 0;

	if(ACTIVE_EXACTMATCH_IDX(s) != NULL) {
		indicies[i++] = ACTIVE_EXACTMATCH_IDX(s);
	}

	if(PENDING_EXACTMATCH_IDX(s) != NULL) {
		indicies[i++] = PENDING_EXACTMATCH_IDX(s);
	}

	if(ACTIVE_FULLTEXT_IDX(s) != NULL) {
		indicies[i++] = ACTIVE_FULLTEXT_IDX(s);
	}

	if(PENDING_FULLTEXT_IDX(s) != NULL) {
		indicies[i++] = PENDING_FULLTEXT_IDX(s);
	}

	return i;
}

// get index from schema
// returns NULL if index wasn't found
Index Schema_GetIndex
(
	const Schema *s,            // schema to get index from
	const Attribute_ID *attrs,  // indexed attributes
	uint n,                     // number of attributes
	IndexType type,             // type of index
	bool include_pending        // take into considiration pending indicies
) {
	// validations
	ASSERT(s != NULL);
	ASSERT((attrs == NULL && n == 0) || (attrs != NULL && n > 0));

	Index idx         = NULL;  // index to return
	uint  idx_count   = 1;     // number of indicies to consider
	Index indicies[4] = {0};   // indicies to consider

	if(type != IDX_ANY) {
		// consider specified index
		if(include_pending) idx_count = 2;
		if(type == IDX_FULLTEXT) {
			indicies[0] = ACTIVE_FULLTEXT_IDX(s);
			if(include_pending) indicies[1] = PENDING_FULLTEXT_IDX(s);
		} else {
			indicies[0] = ACTIVE_EXACTMATCH_IDX(s);
			if(include_pending) indicies[1] = PENDING_EXACTMATCH_IDX(s);
		}
	} else {
		idx_count   = 2;
		indicies[0] = ACTIVE_EXACTMATCH_IDX(s);
		indicies[1] = ACTIVE_FULLTEXT_IDX(s);
		if(include_pending) {
			// consider all indicies exact-match and fulltext indicies
			idx_count   = 4;
			indicies[2] = PENDING_EXACTMATCH_IDX(s);
			indicies[3] = PENDING_FULLTEXT_IDX(s);
		}
	}

	//--------------------------------------------------------------------------
	// return first index which contains all specified attributes
	//--------------------------------------------------------------------------

	for(uint i = 0; i < idx_count; i++) {
		idx = indicies[i];

		// index doesn't exists
		if(idx == NULL) {
			continue;
		}

		// make sure index contains all specified attributes
		bool all_attr_found = true;
		for(uint i = 0; i < n; i++) {
			if(!Index_ContainsAttribute(idx, attrs[i])) {
				idx = NULL;
				all_attr_found = false;
				break;
			}
		}

		if(all_attr_found == true) {
			break;
		}
	}

	return idx;
}

// assign a new index to attribute
// attribute must already exists and not associated with an index
int Schema_AddIndex
(
	Index *idx,         // [input/output] index to create
	Schema *s,          // schema holding the index
	IndexField *field,  // field to index
	IndexType type      // type of entities to index
) {
	ASSERT(s != NULL);
	ASSERT(idx != NULL);
	ASSERT(field != NULL);

	int res;

	if(type == IDX_FULLTEXT) {
		res = Schema_AddFullTextIndex(idx, s, field);
	} else {
		res = Schema_AddExactMatchIndex(idx, s, field);
	}

	return res;
}

int Schema_RemoveIndex
(
	Schema *s,
	const char *field,
	IndexType type
) {
	ASSERT(s != NULL);

	switch(type) {
		case IDX_FULLTEXT:
			return _Schema_RemoveFullTextIndex(s);
		case IDX_EXACT_MATCH:
			return _Schema_RemoveExactMatchIndex(s, field);
		default:
			return INDEX_FAIL;
	}
}

// activate pending exact-match index
// asserts that pending exact-match index is enabled
// drops current active exact-exact index if exists
void Schema_ActivateIndex
(
	Schema *s,  // schema to activate index on
	Index idx   // index to activate
) {
	ASSERT(s != NULL);
	// make sure pending index is enabled
	ASSERT(Index_Enabled(idx) == true);

	Index pending_full_text   = PENDING_FULLTEXT_IDX(s);
	Index pending_exact_match = PENDING_EXACTMATCH_IDX(s);

	// index to activate must be a pending index
	ASSERT(idx == pending_exact_match || idx == pending_full_text);

	if(idx == pending_exact_match) {
		Schema_ActivateExactMatchIndex(s);
	} else {
		Schema_ActivateFullTextIdx(s);
	}
}

// index node under all schema indices
void Schema_AddNodeToIndices
(
	const Schema *s,
	const Node *n
) {
	ASSERT(s != NULL);
	ASSERT(n != NULL);

	Index idx = NULL;

	idx = ACTIVE_EXACTMATCH_IDX(s);
	if(idx != NULL) Index_IndexNode(idx, n);

	idx = PENDING_EXACTMATCH_IDX(s);
	if(idx != NULL) Index_IndexNode(idx, n);

	idx = ACTIVE_FULLTEXT_IDX(s);
	if(idx != NULL) Index_IndexNode(idx, n);

	idx = PENDING_FULLTEXT_IDX(s);
	if(idx != NULL) Index_IndexNode(idx, n);
}

// index edge under all schema indices
void Schema_AddEdgeToIndices
(
	const Schema *s,
	const Edge *e
) {
	ASSERT(s != NULL);
	ASSERT(e != NULL);

	Index idx = NULL;

	idx = ACTIVE_EXACTMATCH_IDX(s);
	if(idx != NULL) Index_IndexEdge(idx, e);

	idx = PENDING_EXACTMATCH_IDX(s);
	if(idx != NULL) Index_IndexEdge(idx, e);
}

// remove node from schema indicies
void Schema_RemoveNodeFromIndices
(
	const Schema *s,
	const Node *n
) {
	ASSERT(s != NULL);
	ASSERT(n != NULL);

	Index idx = NULL;

	idx = ACTIVE_EXACTMATCH_IDX(s);
	if(idx != NULL) Index_RemoveNode(idx, n);

	idx = PENDING_EXACTMATCH_IDX(s);
	if(idx != NULL) Index_RemoveNode(idx, n);

	idx = ACTIVE_FULLTEXT_IDX(s);
	if(idx != NULL) Index_RemoveNode(idx, n);

	idx = PENDING_FULLTEXT_IDX(s);
	if(idx != NULL) Index_RemoveNode(idx, n);
}

// remove edge from schema indicies
void Schema_RemoveEdgeFromIndices
(
	const Schema *s,
	const Edge *e
) {
	ASSERT(s != NULL);
	ASSERT(e != NULL);

	Index idx = NULL;

	idx = ACTIVE_EXACTMATCH_IDX(s);
	if(idx != NULL) Index_RemoveEdge(idx, e);

	idx = PENDING_EXACTMATCH_IDX(s);
	if(idx != NULL) Index_RemoveEdge(idx, e);
}

//------------------------------------------------------------------------------
// constraints API
//------------------------------------------------------------------------------

// check if schema has constraints
bool Schema_HasConstraints
(
	const Schema *s  // schema to query
) {
	ASSERT(s != NULL);
	return (s->constraints != NULL && array_len(s->constraints) > 0);
}

// checks if schema constains constraint
bool Schema_ContainsConstraint
(
	const Schema *s,            // schema to search
	ConstraintType t,           // constraint type
	const Attribute_ID *attrs,  // constraint attributes
	uint attr_count             // number of attributes
) {
	// validations
	ASSERT(s          != NULL);
	ASSERT(attrs      != NULL);
	ASSERT(attr_count > 0);

	Constraint c = Schema_GetConstraint(s, t, attrs, attr_count);
	return (c != NULL && Constraint_GetStatus(c) != CT_FAILED);
}

// retrieves constraint 
// returns NULL if constraint was not found
Constraint Schema_GetConstraint
(
	const Schema *s,            // schema from which to get constraint
	ConstraintType t,           // constraint type
	const Attribute_ID *attrs,  // constraint attributes
	uint attr_count             // number of attributes
) {
	// validations
	ASSERT(s          != NULL);
	ASSERT(attrs      != NULL);
	ASSERT(attr_count > 0);

	// search for constraint
	uint n = array_len(s->constraints);
	for(uint i = 0; i < n; i++) {
		Constraint c = s->constraints[i];

		// check constraint type
		if(Constraint_GetType(c) != t) {
			continue;
		}

		// make sure constraint attribute count matches
		const Attribute_ID *c_attrs;
		uint n = Constraint_GetAttributes(c, &c_attrs, NULL);
		if(n != attr_count) {
			continue;
		}

		// match each attribute
		bool match = true;  // optimistic
		for(uint j = 0; j < n; j++) {
			if(c_attrs[j] != attrs[j]) {
				match = false;
				break;
			}
		}

		if(match == true) {
			return c;
		}
	}

	// no constraint was found
	return NULL;
}

// get all constraints in schema
const Constraint *Schema_GetConstraints
(
	const Schema *s  // schema from which to extract constraints
) {
	// validations
	ASSERT(s != NULL);
	ASSERT(s->constraints != NULL);

	return s->constraints;
}

// adds a constraint to schema
void Schema_AddConstraint
(
	Schema *s,       // schema holding the index
	Constraint c     // constraint to add
) {
	ASSERT(s != NULL);
	ASSERT(c != NULL);
	array_append(s->constraints, c);
}

// removes constraint from schema
void Schema_RemoveConstraint
(
	Schema *s,    // schema
	Constraint c  // constraint to remove
) {
	// validations
	ASSERT(s != NULL);
	ASSERT(c != NULL);

	// search for constraint
	uint n = array_len(s->constraints);
	for(uint i = 0; i < n; i++) {
		if(c == s->constraints[i]) {
			Constraint_IncPendingChanges(c);
			array_del_fast(s->constraints, i);
			return;
		}
	}

	ASSERT(false);
}

// enforce all constraints under given schema on entity
bool Schema_EnforceConstraints
(
	const Schema *s,       // schema
	const GraphEntity *e,  // entity to enforce
	char **err_msg         // report error message
) {
	// validations
	ASSERT(s != NULL);
	ASSERT(e != NULL);
	
	// see if entity holds under all schema's constraints
	uint n = array_len(s->constraints);
	for(uint i = 0; i < n; i++) {
		Constraint c = s->constraints[i];
		if(Constraint_GetStatus(c) != CT_FAILED &&
		   !Constraint_EnforceEntity(c, e, err_msg)) {
			// entity failed to pass constraint
			return false;
		}
	}

	// entity passed all constraint
	return true;
}

void Schema_Free
(
	Schema *s
) {
	ASSERT(s != NULL);

	if(s->name) {
		rm_free(s->name);
	}

	// free constraints
	if(s->constraints != NULL) {
		uint n = array_len(s->constraints);
		for(uint i = 0; i < n; i++) {
			Constraint_Free(s->constraints + i);
		}
		array_free(s->constraints);
	}

	// free indicies
	if(PENDING_FULLTEXT_IDX(s) != NULL) {
		Index_Free(PENDING_FULLTEXT_IDX(s));
	}

	if(ACTIVE_FULLTEXT_IDX(s) != NULL) {
		Index_Free(ACTIVE_FULLTEXT_IDX(s));
	}

	if(PENDING_EXACTMATCH_IDX(s)) {
		Index_Free(PENDING_EXACTMATCH_IDX(s));
	}

	if(ACTIVE_EXACTMATCH_IDX(s) != NULL) {
		Index_Free(ACTIVE_EXACTMATCH_IDX(s));
	}

	rm_free(s);
}

