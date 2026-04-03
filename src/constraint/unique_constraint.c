/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "constraint.h"
#include "../query_ctx.h"
#include "../index/index.h"
#include "redisearch_api.h"
#include "../src/datatypes/point.h"
#include "../graph/entities/attribute_set.h"

#include <stdatomic.h>

// opaque structure representing a constraint
struct _UniqueConstraint {
	uint8_t n_attr;                         // number of fields
	ConstraintType t;                       // constraint type
	Constraint_EnforcementCB enforce;       // enforcement function
	Constraint_SetPrivateDataCB set_pdata;  // set private data
	Constraint_GetPrivateDataCB get_pdata;  // get private data
	int schema_id;                          // enforced schema ID
	AttributeID *attrs;                     // enforced attributes
	const char **attr_names;                // enforced attribute names
	ConstraintStatus status;                // constraint status
	uint _Atomic pending_changes;           // number of pending changes
	GraphEntityType et;                     // entity type
	Index idx;                              // supporting index
};

typedef struct _UniqueConstraint* UniqueConstraint;

static const char *_node_violation_err_msg =
	EMSG_UNIQUE_CONSTRAINT_VIOLATION_NODE;

static const char *_edge_violation_err_msg =
	EMSG_UNIQUE_CONSTRAINT_VIOLATION_EDGE;

// sets constraint private data
static void _SetPrivateData
(
	Constraint c,  // constraint to update
	void *pdata    // private data
) {
	ASSERT(c != NULL);
	ASSERT(pdata != NULL);

	UniqueConstraint _c = (UniqueConstraint)c;
	_c->idx = (Index)pdata;
}

// gets constraint private data
static void* _GetPrivateData
(
	Constraint c
) {
	ASSERT(c != NULL);

	UniqueConstraint _c = (UniqueConstraint)c;
	return _c->idx;
}

// enforces unique constraint on given entity
// returns true if entity confirms with constraint false otherwise
bool EnforceUniqueEntity
(
	const Constraint c,    // constraint to enforce
	const GraphEntity *e,  // enforced entity
	char **err_msg         // report error message
) {
	// validations
	ASSERT (c != NULL) ;
	ASSERT (e != NULL) ;

	UniqueConstraint _c = (UniqueConstraint)c ;

	//--------------------------------------------------------------------------
	// validate entity has all required attributes
	//--------------------------------------------------------------------------

	const AttributeSet attributes = GraphEntity_GetAttributes (e) ;
	SIValue attrs[_c->n_attr] ;

	for (uint8_t i = 0; i < _c->n_attr; i++) {
		AttributeID attr_id = _c->attrs[i] ;

		// make sure entity possesses attribute
		if (!AttributeSet_Get (attributes, attr_id, attrs + i)) {
			// entity satisfies constraint in a vacuous truth manner
			return true;
		}

		// validate attribute type
		SIType t = SI_TYPE (attrs[i]) ;
		if (t & ~(T_STRING | T_BOOL | SI_NUMERIC)) {
			// TODO: see RediSearch MULTI-VALUE index
			// TODO: RediSearch exact match for point
			return true ;
		}
	}

	//--------------------------------------------------------------------------
	// query RediSearch index
	//--------------------------------------------------------------------------

	// construct a unique constraint query tree
	// TODO: prefer to have the RediSearch query "template" constructed
	// once and reused for each entity
	Index idx = _c->idx;
	RSQNode *root = Index_BuildUniqueConstraintQuery (idx, attrs, _c->attrs,
			_c->n_attr);

	bool holds = false;  // return value none-optimistic

	// constraint holds if there are no duplicates, a single index match
	RSIndex *rs_idx = Index_RSIndex(idx);
	RSResultsIterator *iter = RediSearch_GetResultsIterator(root, rs_idx);
	if(Constraint_GetEntityType(c) == GETYPE_NODE) {
		// first call, expecting to find 'e' in the index
		const EntityID *id =
			(EntityID*)RediSearch_ResultsIteratorNext(iter, rs_idx, NULL);

		ASSERT(id != NULL);

		if(*id != ENTITY_GET_ID(e)) {
			holds = false;
			goto cleanup;
		}
	} else {
		// first call, expecting to find 'e' in the index
		const EdgeIndexKey *id =
			(EdgeIndexKey*)RediSearch_ResultsIteratorNext(iter, rs_idx, NULL);

		ASSERT(id != NULL);

		if(id->edge_id != ENTITY_GET_ID(e)) {
			holds = false;
			goto cleanup;
		}
	}

	// second call, holds if no value is returned
	holds = RediSearch_ResultsIteratorNext(iter, rs_idx, NULL) == NULL;

cleanup:
	RediSearch_ResultsIteratorFree(iter);

	if(holds == false && err_msg != NULL) {
		int res;
		UNUSED(res);
		// entity violates constraint, compose error message
		GraphContext *gc = QueryCtx_GetGraphCtx();
		SchemaType st = (_c->et == GETYPE_NODE) ? SCHEMA_NODE : SCHEMA_EDGE;
		Schema *s = GraphContext_GetSchemaByID(gc, _c->schema_id, st);
		if(Constraint_GetEntityType(c) == GETYPE_NODE) {
			res = asprintf(err_msg, _node_violation_err_msg, Schema_GetName(s));
		} else {
			res = asprintf(err_msg, _edge_violation_err_msg, Schema_GetName(s));
		}
	}

	return holds;
}

Constraint Constraint_UniqueNew
(
	int schema_id,            // schema ID
	AttributeID *fields,     // enforced fields
	const char **attr_names,  // enforced attribute names
	uint8_t n_fields,         // number of fields
	GraphEntityType et,       // entity type
	Index idx                 // index
) {
	UniqueConstraint c = rm_malloc(sizeof(struct _UniqueConstraint));

	// introduce constraint attributes
	c->attrs = rm_malloc(sizeof(AttributeID) * n_fields);
	memcpy(c->attrs, fields, sizeof(AttributeID) * n_fields);

	c->attr_names = rm_malloc(sizeof(char*) * n_fields);
	memcpy(c->attr_names, attr_names, sizeof(char*) * n_fields);

	// initialize constraint
	c->t               = CT_UNIQUE;
	c->et              = et;
	c->idx             = idx;
	c->n_attr          = n_fields;
	c->status          = CT_PENDING;
	c->enforce         = EnforceUniqueEntity;
	c->set_pdata       = _SetPrivateData;
	c->get_pdata       = _GetPrivateData;
	c->schema_id       = schema_id;
	c->pending_changes = ATOMIC_VAR_INIT(0);

	return (Constraint)c;
}

