/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "value.h"
#include "constraint.h"
#include "../util/arr.h"
#include "../index/indexer.h"
#include "../graph/graphcontext.h"
#include "../graph/entities/attribute_set.h"
#include "../graph/delta_matrix/delta_matrix_iter.h"

#include <stdatomic.h>

// opaque structure representing a constraint
typedef struct _Constraint {
	uint8_t n_attr;                         // number of fields
	ConstraintType t;                       // constraint type
	Constraint_EnforcementCB enforce;       // enforcement function
	Constraint_SetPrivateDataCB set_pdata;  // set private data
	Constraint_GetPrivateDataCB get_pdata;  // get private data
	int schema_id;                          // enforced label/relationship-type
	AttributeID *attrs;                     // enforced attributes
	const char **attr_names;                // enforced attribute names
	ConstraintStatus status;                // constraint status
	uint _Atomic pending_changes;           // number of pending changes
	GraphEntityType et;                     // entity type
} _Constraint;

// Extern functions

// create a new unique constraint
extern Constraint Constraint_UniqueNew
(
	LabelID l,                // label/relation ID
	AttributeID *fields,      // enforced fields
	const char **attr_names,  // enforced attribute names
	uint8_t n_fields,         // number of fields
	GraphEntityType et,       // entity type
	Index idx                 // index
);

// create a new mandatory constraint
extern Constraint Constraint_MandatoryNew
(
	LabelID l,                // label/relation ID
	AttributeID *fields,      // enforced fields
	const char **attr_names,  // enforced attribute names
	uint8_t n_fields,         // number of fields
	GraphEntityType et        // entity type
);

// enforces unique constraint on given entity
// returns true if entity confirms with constraint false otherwise
extern bool EnforceUniqueEntity
(
	const Constraint c,    // constraint to enforce
	const GraphEntity *e,  // enforced entity
	char **err_msg         // report error message
);

// enforces mandatory constraint on given entity
extern bool Constraint_EnforceMandatory
(
	const Constraint c,    // constraint to enforce
	const GraphEntity *e,  // enforced entity
	char **err_msg         // report error message
);

// disabled constraint enforce function
// simply returns true
static bool Constraint_EnforceNOP
(
	const Constraint c,    // constraint to enforce
	const GraphEntity *e,  // enforced entity
	char **err_msg         // report error message
) {
	return true;
}

// create a new constraint
Constraint Constraint_New
(
	struct GraphContext *gc,
	ConstraintType t,         // type of constraint
	int schema_id,            // schema ID
	AttributeID *fields,      // enforced fields
	const char **attr_names,  // enforced attribute names
	uint8_t n_fields,         // number of fields
	GraphEntityType et,       // entity type
	const char **err_msg      // error message
) {
	ASSERT(t == CT_UNIQUE || t == CT_MANDATORY);

	Constraint c = NULL;

	if(t == CT_UNIQUE) {
		// a unique constraints requires an index, try to get supporting index
		SchemaType st = (et == GETYPE_NODE) ? SCHEMA_NODE : SCHEMA_EDGE;
		Schema *s = GraphContext_GetSchemaByID((GraphContext*)gc, schema_id, st);
		Index idx = Schema_GetIndex(s, fields, n_fields, INDEX_FLD_RANGE, true);

		// supporting index is missing, can't create constraint
		if(idx == NULL) {
			if(err_msg != NULL) {
				*err_msg = "missing supporting exact-match index";
			}
			return NULL;
		}

		// create a new unique constraint
		c = Constraint_UniqueNew(schema_id, fields, attr_names, n_fields, et,
				idx);
	} else {
		// create a new mandatory constraint
		c = Constraint_MandatoryNew(schema_id, fields, attr_names, n_fields, et);
	}

	ASSERT(Constraint_GetStatus(c) == CT_PENDING);

	return c;
}

// enable constraint
void Constraint_Enable
(
	Constraint c  // constraint to enable
) {
	ASSERT(c != NULL);
	switch(Constraint_GetType(c)) {
		case CT_UNIQUE:
			c->enforce = EnforceUniqueEntity;
			break;
		case CT_MANDATORY:
			c->enforce = Constraint_EnforceMandatory;
			break;
	}
}

// disable constraint
void Constraint_Disable
(
	Constraint c  // constraint to disable
) {
	ASSERT(c != NULL);

	c->enforce = Constraint_EnforceNOP;
}

// returns constraint's type
ConstraintType Constraint_GetType
(
	const Constraint c  // constraint to query
) {
	ASSERT(c != NULL);
	return c->t;
}

// returns constraint entity type
GraphEntityType Constraint_GetEntityType
(
	const Constraint c  // constraint to query
) {
	ASSERT(c != NULL);

	return c->et;
}

// returns constraint schema ID
int Constraint_GetSchemaID
(
	const Constraint c  // constraint to query
) {
	ASSERT(c != NULL);

	return c->schema_id;
}

// returns constraint status
ConstraintStatus Constraint_GetStatus
(
	const Constraint c
) {
	ASSERT(c != NULL);
	return c->status;
}

// set constraint status
// status can change from:
// 1. CT_PENDING to CT_ACTIVE
// 2. CT_PENDING to CT_FAILED
void Constraint_SetStatus
(
	Constraint c,            // constraint to update
	ConstraintStatus status  // new status
) {
	// validations
	// validate state transition
	ASSERT(c->status == CT_PENDING);
	ASSERT(status == CT_ACTIVE || status == CT_FAILED);

	// assuming under lock
    c->status = status;
}

// sets constraint private data
void Constraint_SetPrivateData
(
	Constraint c,  // constraint to update
	void *pdata    // private data
) {
	ASSERT(c != NULL);

	if(c->set_pdata != NULL) {
		c->set_pdata(c, pdata);
	}
}

// get constraint private data
void *Constraint_GetPrivateData
(
	Constraint c  // constraint from which to get private data
) {
	ASSERT(c != NULL);

	if(c->get_pdata != NULL) {
		return c->get_pdata(c);
	}

	return NULL;
}

// returns a shallow copy of constraint attributes
uint8_t Constraint_GetAttributes
(
	const Constraint c,            // constraint from which to get attributes
	const AttributeID **attr_ids,  // array of constraint attribute IDs
	const char ***attr_names       // array of constraint attribute names
) {
	ASSERT(c != NULL);

	if(attr_ids != NULL) {
		*attr_ids = c->attrs;
	}

	if(attr_names != NULL) {
		*attr_names = c->attr_names;
	}

	return c->n_attr;
}

// checks if constraint enforces attribute
bool Constraint_ContainsAttribute
(
	Constraint c,        // constraint to query
	AttributeID attr_id  // enforced attribute
) {
	for(uint8_t i = 0; i < c->n_attr; i++) {
		if(c->attrs[i] == attr_id) {
			return true;
		}
    }

    return false;
}

// returns number of pending changes
int Constraint_PendingChanges
(
	const Constraint c  // constraint to inquery
) {
	// validations
	ASSERT(c != NULL);

	// the number of pending changes of a constraint can not be more than 2
	// CREATE + DROP will yield 2 pending changes, a dropped constraint can not
	// be dropped again
	//
	// CREATE + CREATE will result in two different constraints each with its
	// own pending changes
    ASSERT(c->pending_changes <= 2);

	return c->pending_changes;
}

// increment number of pending changes
void Constraint_IncPendingChanges
(
	Constraint c  // constraint to update
) {
	ASSERT(c != NULL);

	// one can't create or drop the same constraint twice
	// see comment at Constraint_PendingChanges
    ASSERT(c->pending_changes < 2);

	// atomic increment
	c->pending_changes++;
}

// decrement number of pending changes
void Constraint_DecPendingChanges
(
	Constraint c  // constraint to update
) {
	ASSERT(c != NULL);

	// one can't create or drop the same constraint twice
	// see comment at Constraint_PendingChanges
    ASSERT(c->pending_changes > 0);
    ASSERT(c->pending_changes <= 2);

	// atomic decrement
	c->pending_changes--;
}

// replicate constraint to both persistency and replicas
void Constraint_Replicate
(
	RedisModuleCtx *ctx,           // redis module context
	const Constraint c,            // constraint to replicate
	const struct GraphContext *gc  // graph context
) {
	// CREATE <key> UNIQUE/MANDATORY [NODE label / RELATIONSHIP type] PROPERTIES prop_count prop0, prop1...

	// command format
	// 1. c - CREATE
	// 2. c - graph_name
	// 3. c - constraint type
	// 4. c - entity type NODE/RELATIONSHIP
	// 5. c - label
	// 6. c - PROPERTIES
	// 7. l - #props
	// 8. v - [prop0, prop1, ...]
	char *fmt = "cccccclv";

	// graph name
	GraphContext *_gc = (GraphContext*)gc;
	const char *graph_name = GraphContext_GetName(_gc);

	// constraint type
	const char *c_type = (Constraint_GetType(c) == CT_UNIQUE) // constraint type
		? "UNIQUE"
		: "MANDATORY";

	// entity type
	char *et;
	SchemaType st;
	if(Constraint_GetEntityType(c) == GETYPE_NODE) {
		et = "NODE";
		st = SCHEMA_NODE;
	} else {
		et = "RELATIONSHIP";
		st = SCHEMA_EDGE;
	}

	// label
	Schema *s = GraphContext_GetSchemaByID(_gc, Constraint_GetSchemaID(c), st);
	const char *label = Schema_GetName(s);

	// properties
	RedisModuleString *attrs[c->n_attr];
	for(uint i = 0; i < c->n_attr; i++) {
		const char *attr = c->attr_names[i];
		attrs[i] = RedisModule_CreateString(ctx, attr, strlen(attr));
	}

	// replicate
	RedisModule_Replicate(ctx, "GRAPH.CONSTRAINT", fmt, "CREATE", graph_name,
			c_type, et, label, "PROPERTIES", c->n_attr, attrs, (size_t)c->n_attr);

	// free strings
	for(uint i = 0; i < c->n_attr; i++) {
		RedisModule_FreeString(ctx, attrs[i]);
	}
}

// tries to enforce constraint
void Constraint_Enforce
(
	Constraint c,            // constraint to enforce
	struct GraphContext *gc  // graph context
) {
	ASSERT(c != NULL);
	ASSERT(Constraint_GetStatus(c) == CT_PENDING);

	// mark constraint as pending
	Constraint_IncPendingChanges(c);

	// add constraint enforcement task
	Indexer_EnforceConstraint(c, (GraphContext*)gc);
}

// enforce constraint on all relevant nodes
void Constraint_EnforceNodes
(
	Constraint c,
	Graph *g
) {
	// check if constraint holds
	// scan through all entities governed by this constraint and enforce

	ASSERT(c != NULL);
	ASSERT(g != NULL);

	
	Delta_MatrixTupleIter it;                         // matrix iterator
	bool                  holds      = true;          // constraint holds
	GrB_Index             rowIdx     = 0;             // current row being scanned
	int                   enforced   = 0;             // #entities in current batch
	int                   schema_id  = c->schema_id;  // constraint schema ID
	int                   batch_size = 10000;         // #entities to enforce

	while(holds) {
		// lock graph for reading
		Graph_AcquireReadLock(g);

		// constraint state changed, abort enforcement
		// this can happen if for example the following sequance is issued:
		// 1. CREATE CONSTRAINT...
		// 2. DROP CONSTRAINT...
		if(Constraint_PendingChanges(c) > 1) {
			break;
		}

		const Delta_Matrix m = Graph_GetLabelMatrix(g, schema_id);
		ASSERT(m != NULL);

		// reset number of enforce nodes in batch
		enforced = 0;

		//----------------------------------------------------------------------
		// resume scanning from rowIdx
		//----------------------------------------------------------------------

		GrB_Info info;
		info = Delta_MatrixTupleIter_AttachRange(&it, m, rowIdx, UINT64_MAX);
		ASSERT(info == GrB_SUCCESS);

		//----------------------------------------------------------------------
		// batch enforce nodes
		//----------------------------------------------------------------------

		EntityID id;
		while(enforced < batch_size &&
			  Delta_MatrixTupleIter_next_BOOL(&it, &id, NULL, NULL) == GrB_SUCCESS)
		{
			Node n;
			Graph_GetNode(g, id, &n);
			if(!c->enforce(c, (GraphEntity*)&n, NULL)) {
				// found node which violate constraint
				holds = false;
				break;
			}
			enforced++;
		}

		//----------------------------------------------------------------------
		// done with current batch
		//----------------------------------------------------------------------

		if(enforced != batch_size) {
			// iterator depleted, no more nodes to enforce
			break;
		} else {
			// release read lock
			Graph_ReleaseLock(g);

			// finished current batch
			Delta_MatrixTupleIter_detach(&it);

			// continue next batch from row id+1
			// this is true because we're iterating over a diagonal matrix
			rowIdx = id + 1;
		}
	}

	Delta_MatrixTupleIter_detach(&it);

	// update constraint status
	ConstraintStatus status = (holds) ? CT_ACTIVE : CT_FAILED;
	Constraint_SetStatus(c, status);

	// release read lock
	Graph_ReleaseLock(g);
}

// enforce constraint on all relevant edges
void Constraint_EnforceEdges
(
	Constraint c,
	Graph *g
) {
	bool info;
	TensorIterator it = {0};

	bool      holds        = true;               // constraint holds
	EntityID  src_id       = 0;                  // current processed row idx
	EntityID  dest_id      = 0;                  // current processed column idx
	EntityID  edge_id      = 0;                  // current processed edge idx
	EntityID  prev_src_id  = INVALID_ENTITY_ID;  // last processed row idx
	EntityID  prev_dest_id = INVALID_ENTITY_ID;  // last processed column idx
	int       enforced     = 0;                  // # entities enforced in batch
	int       schema_id    = c->schema_id;       // edge relationship type ID
	int       batch_size   = 1000;               // max number of entities to enforce

	while(holds) {
		// lock graph for reading
		Graph_AcquireReadLock(g);

		// constraint state changed, abort enforcement
		// this can happen if for example the following sequance is issued:
		// 1. CREATE CONSTRAINT...
		// 2. DELETE CONSTRAINT...
		if(Constraint_PendingChanges(c) > 1) {
			break;
		}

		// reset number of enforced edges in batch
		enforced = 0;

		ASSERT(Graph_GetMatrixPolicy(g) == SYNC_POLICY_FLUSH_RESIZE);
		// sync relation matrix
		Delta_Matrix R = Graph_GetRelationMatrix(g, schema_id, false);

		//----------------------------------------------------------------------
		// resume scanning from previous row/col indices
		//----------------------------------------------------------------------

		TensorIterator_ScanRange(&it, R, src_id, UINT64_MAX, false);

		// skip previously enforced edges
		while((info =
				TensorIterator_next(&it, &src_id, &dest_id, &edge_id, NULL)) &&
				src_id == prev_src_id &&
				dest_id < prev_dest_id);

		// process only if iterator is on an active entry
		if(!info) {
			break;
		}

		//----------------------------------------------------------------------
		// batch enforce edges
		//----------------------------------------------------------------------

		do {
			Edge e;
			e.src_id     = src_id;
			e.dest_id    = dest_id;
			e.relationID = schema_id;

			bool res = Graph_GetEdge(g, edge_id, &e);
			assert(res == true);
			if(!c->enforce(c, (GraphEntity*)&e, NULL)) {
				holds = false;
				break;
			}

			if(prev_src_id != src_id || prev_dest_id != dest_id) {
				enforced++;
			}
			prev_src_id  = src_id;
			prev_dest_id = dest_id;
		} while((enforced < batch_size ||
				(prev_src_id == src_id && prev_dest_id == dest_id)) &&
			  TensorIterator_next(&it, &src_id, &dest_id, &edge_id, NULL) &&
			  holds);

		//----------------------------------------------------------------------
		// done with current batch
		//----------------------------------------------------------------------

		if(enforced != batch_size) {
			// iterator depleted, no more edges to enforce
			break;
		} else {
			// finished current batch
			// release read lock
			Graph_ReleaseLock(g);
		}
	}

	// update constraint status
	ConstraintStatus status = (holds) ? CT_ACTIVE : CT_FAILED;
	Constraint_SetStatus(c, status);

	// release read lock
	Graph_ReleaseLock(g);
}

// enforce constraint on entity
// returns true if entity satisfies the constraint
// false otherwise
bool Constraint_EnforceEntity
(
	Constraint c,          // constraint to enforce
	const GraphEntity *e,  // enforced entity
	char **err_msg         // report error message
) {
	ASSERT(c != NULL);
	ASSERT(e != NULL);

	return c->enforce(c, e, err_msg);
}

void Constraint_Free
(
	Constraint *c
) {
	ASSERT(c != NULL);

	Constraint _c = *c;

    rm_free(_c->attrs);
	rm_free(_c->attr_names);
    rm_free(_c);

	*c = NULL;
}
