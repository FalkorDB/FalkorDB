/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "constraint.h"
#include "../errors/errors.h"
#include "../query_ctx.h"
#include "../index/index.h"
#include "../index/index_doc_key.h"
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

	// fail fast if the query already exhausted its time budget: we cannot
	// confirm uniqueness, so abort with a timeout rather than a false violation
	if(QueryCtx_TimedOut()) {
		QueryCtx_SetStatusTimedOut();
		if(err_msg != NULL) {
			int res = asprintf(err_msg, "%s", EMSG_QUERY_TIMEOUT);
			UNUSED(res);
		}
		return false;
	}

	// construct a unique constraint query tree
	// TODO: prefer to have the RediSearch query "template" constructed
	// once and reused for each entity
	Index idx = _c->idx;
	RSQNode *root = Index_BuildUniqueConstraintQuery (idx, attrs, _c->attrs,
			_c->n_attr);

	bool holds     = false;  // return value none-optimistic
	bool timed_out = false;  // iterator hit the query deadline

	// A live constraint keeps its backing index pinned -- DROP INDEX is rejected
	// while a constraint depends on it -- so the strong ref is always valid.
	RSIndex *rs_idx = Index_AcquireRSIndex(idx);
	ASSERT(rs_idx != NULL);

	// constraint holds if there are no duplicates, a single index match
	RSResultsIterator *iter = RediSearch_GetResultsIteratorWithTimeout(root,
			rs_idx, QueryCtx_GetRemainingTimeMS());
	if(Constraint_GetEntityType(c) == GETYPE_NODE) {
		// first call, expecting to find 'e' in the index
		size_t len = 0;
		const char *doc_key =
			(const char *)RediSearch_ResultsIteratorNext(iter, rs_idx, &len);

		// NULL means the iterator either timed out or matched no docs. On a
		// timeout we cannot confirm uniqueness, so flag it and abort as a
		// timeout in cleanup; otherwise refuse the entity (consistent with the
		// second-call branch).
		if(doc_key == NULL) {
			timed_out = RediSearch_ResultsIteratorTimedOut(iter);
			holds = false;
			goto cleanup;
		}

		EntityID id;
		if(!IndexDocKey_DecodeNode(doc_key, len, &id)) {
			holds = false;
			goto cleanup;
		}

		if(id != ENTITY_GET_ID(e)) {
			holds = false;
			goto cleanup;
		}
	} else {
		// first call, expecting to find 'e' in the index
		size_t len = 0;
		const char *doc_key =
			(const char *)RediSearch_ResultsIteratorNext(iter, rs_idx, &len);

		// see node branch above for the NULL / timeout rationale
		if(doc_key == NULL) {
			timed_out = RediSearch_ResultsIteratorTimedOut(iter);
			holds = false;
			goto cleanup;
		}

		EdgeIndexKey id;
		if(!IndexDocKey_DecodeEdge(doc_key, len, &id)) {
			holds = false;
			goto cleanup;
		}

		if(id.edge_id != ENTITY_GET_ID(e)) {
			holds = false;
			goto cleanup;
		}
	}

	// second call: a NULL means either "no duplicate" (holds) or a timeout --
	// a timeout must not be read as uniqueness confirmed
	if(RediSearch_ResultsIteratorNext(iter, rs_idx, NULL) == NULL) {
		timed_out = RediSearch_ResultsIteratorTimedOut(iter);
		holds     = !timed_out;
	} else {
		holds = false;  // a second match => duplicate => violation
	}

cleanup:
	RediSearch_ResultsIteratorFree(iter);
	Index_ReleaseRSIndex(rs_idx);

	// timeout takes precedence over a (possibly spurious) violation: we could
	// not confirm uniqueness, so report the timeout rather than reject the write
	if(timed_out) {
		QueryCtx_SetStatusTimedOut();
		if(err_msg != NULL) {
			int res = asprintf(err_msg, "%s", EMSG_QUERY_TIMEOUT);
			UNUSED(res);
		}
		return false;
	}

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

