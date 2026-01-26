/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "schema.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../errors/errors.h"
#include "../index/indexer.h"
#include "../graph/graphcontext.h"
#include "../constraint/constraint.h"

// activate pending index
// asserts that pending index is enabled
// drops current active index if exists
void Schema_ActivateIndex
(
	Schema *s   // schema to activate index on
) {
	Index active  = ACTIVE_IDX(s);
	Index pending = PENDING_IDX(s);

	ASSERT(Index_Enabled(pending) == true);

	// drop active if exists
	if(active != NULL) {
		Index_Free(active);
	}

	// set pending index as active
	ACTIVE_IDX(s) = pending;

	// clear pending index
	PENDING_IDX(s) = NULL;

	//--------------------------------------------------------------------------
	// update unique constraint private data
	//--------------------------------------------------------------------------

	active = ACTIVE_IDX(s);

	// uniqie constraint rely on indicies
	// whenever such an index is updated
	// we need to update the relevant uniqie constraint
	uint n = array_len(s->constraints);
	for(uint i = 0; i < n; i++) {
		Constraint c = s->constraints[i];
		Constraint_SetPrivateData(c, active);
	}
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

	return (ACTIVE_IDX(s) || PENDING_IDX(s));
}

// retrieves all indicies from schema
// active index
// pending index
// returns number of indicies set
unsigned short Schema_GetIndicies
(
	const Schema *s,
	Index indicies[2]
) {
	int i = 0;

	if(ACTIVE_IDX(s) != NULL) {
		indicies[i++] = ACTIVE_IDX(s);
	}

	if(PENDING_IDX(s) != NULL) {
		indicies[i++] = PENDING_IDX(s);
	}

	return i;
}

// get index from schema
// returns NULL if index wasn't found
Index Schema_GetIndex
(
	const Schema *s,           // schema to get index from
	const AttributeID *attrs,  // indexed attributes
	uint n,                    // number of attributes
	IndexFieldType t,          // all index attributes must be of this type
	bool include_pending       // take into considiration pending indicies
) {
	// validations
	ASSERT(s != NULL);
	ASSERT((attrs == NULL && n == 0) || (attrs != NULL && n > 0));

	Index idx         = NULL;  // index to return
	uint  idx_count   = 1;     // number of indicies to consider
	Index indicies[2] = {0};   // indicies to consider

	indicies[0] = ACTIVE_IDX(s);
	if(include_pending) {
		idx_count   = 2;
		indicies[1] = PENDING_IDX(s);
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
			if(!Index_ContainsField(idx, attrs[i], t)) {
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

// assign a new attribute to index
// attribute must not be associated with an index
int Schema_AddIndex
(
	Index *idx,        // [input/output] index to create
	Schema *s,         // schema holding the index
	IndexField *field  // field to index
) {
	ASSERT(s     != NULL);
	ASSERT(idx   != NULL);
	ASSERT(field != NULL);

	*idx = NULL;  // set default

	Index _idx = NULL;
	GraphEntityType et = (s->type == SCHEMA_NODE) ? GETYPE_NODE : GETYPE_EDGE;

	//--------------------------------------------------------------------------
	// determine which index to populate
	//--------------------------------------------------------------------------

	// if pending-index exists, reuse it
	// if active-index exists, clone it and use clone
	// otherwise (first index) create a new index
	Index active  = ACTIVE_IDX(s);
	Index pending = PENDING_IDX(s);
	Index altered = (pending != NULL) ? pending : active;

	// see if field is already indexed
	if(altered != NULL) {
		// error if field is already indexed
		if(Index_ContainsField(altered, field->id, field->type)) {
			// field already indexed, error and return
			ErrorCtx_SetError(EMSG_INDEX_FIELD_ALREADY_EXISTS, field->name);
			IndexField_Free(field);
			return INDEX_FAIL;
		}
	}

	_idx = pending;
	if(pending == NULL) {
		if(active != NULL) {
			_idx = Index_Clone(active);
		} else {
			_idx = Index_New(s->name, s->id, et);
		}
	}
	PENDING_IDX(s) = _idx;  // set pending exact-match index

	// add field to index
	int res = Index_AddField(_idx, field);
	ASSERT(res == INDEX_OK);

	*idx = _idx;
	return res;
}

int Schema_RemoveIndex
(
	Schema *s,        // schema to remove index from
	const char *f,    // field to remove from index
	IndexFieldType t  // field type
) {
	ASSERT(s != NULL);
	ASSERT(f != NULL);
	ASSERT(t & (INDEX_FLD_RANGE | INDEX_FLD_FULLTEXT | INDEX_FLD_VECTOR));

	GraphContext *gc = QueryCtx_GetGraphCtx();

	// convert attribute name to attribute ID
	AttributeID attr_id = GraphContext_GetAttributeID(gc, f);
	if(attr_id == ATTRIBUTE_ID_NONE) {
		return INDEX_FAIL;
	}

	// try to get index
	// if a pending index exists use it otherwise use the active index
	Index active  = ACTIVE_IDX(s);
	Index pending = PENDING_IDX(s);
	Index idx     = pending;

	// both pending and active indicies do not exists
	if(pending == NULL) {
		if(active == NULL) {
			return INDEX_FAIL;
		}
		// use active
		pending = Index_Clone(active);
		PENDING_IDX(s) = pending;
		idx = pending;
	}

	// index doesn't containts attribute
	if(Index_ContainsField(idx, attr_id, t) == false) {
		return INDEX_FAIL;
	}

	//--------------------------------------------------------------------------
	// make sure index doesn't supports any constraints
	//--------------------------------------------------------------------------

	if(t == INDEX_FLD_RANGE) {
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
	}

	Index_RemoveField(idx, attr_id, t);

	// if index field count dropped to 0 remove index from schema
	// index will be freed by the indexer thread
	if(Index_FieldsCount(idx) == 0) {
		if(active != NULL) {
			ACTIVE_IDX(s) = NULL;  // disconnect index from schema
			Indexer_DropIndex(active, gc);
		}

		if(pending != NULL) {
			PENDING_IDX(s) = NULL;  // disconnect index from schema
			Indexer_DropIndex(pending, gc);
		}
	} else {
		Indexer_PopulateIndex(gc, s, idx);
	}

	return INDEX_OK;
}

// index node under all schema index
void Schema_AddNodeToIndex
(
	const Schema *s,
	const Node *n
) {
	ASSERT (s != NULL) ;
	ASSERT (n != NULL) ;

	Index idx = NULL ;

	idx = ACTIVE_IDX (s) ;
	if (idx != NULL) {
		Index_IndexNode (idx, n) ;
	}

	idx = PENDING_IDX (s) ;
	if (idx != NULL) {
		Index_IndexNode (idx, n) ;
	}
}

// index edge under all schema index
void Schema_AddEdgeToIndex
(
	const Schema *s,
	const Edge *e
) {
	ASSERT(s != NULL);
	ASSERT(e != NULL);

	Index idx = NULL;

	idx = ACTIVE_IDX(s);
	if(idx != NULL) Index_IndexEdge(idx, e);

	idx = PENDING_IDX(s);
	if(idx != NULL) Index_IndexEdge(idx, e);
}

// remove node from schema index
void Schema_RemoveNodeFromIndex
(
	const Schema *s,
	const Node *n
) {
	ASSERT(s != NULL);
	ASSERT(n != NULL);

	Index idx = NULL;

	idx = ACTIVE_IDX(s);
	if(idx != NULL) Index_RemoveNode(idx, n);

	idx = PENDING_IDX(s);
	if(idx != NULL) Index_RemoveNode(idx, n);
}

// remove edge from schema index
void Schema_RemoveEdgeFromIndex
(
	const Schema *s,
	const Edge *e
) {
	ASSERT(s != NULL);
	ASSERT(e != NULL);

	Index idx = NULL;

	idx = ACTIVE_IDX(s);
	if(idx != NULL) Index_RemoveEdge(idx, e);

	idx = PENDING_IDX(s);
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
	const Schema *s,           // schema to search
	ConstraintType t,          // constraint type
	const AttributeID *attrs,  // constraint attributes
	uint attr_count            // number of attributes
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
	const Schema *s,           // schema from which to get constraint
	ConstraintType t,          // constraint type
	const AttributeID *attrs,  // constraint attributes
	uint attr_count            // number of attributes
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
		const AttributeID *c_attrs;
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
	if(PENDING_IDX(s) != NULL) {
		Index_Free(PENDING_IDX(s));
	}

	if(ACTIVE_IDX(s) != NULL) {
		Index_Free(ACTIVE_IDX(s));
	}

	rm_free(s);
}

