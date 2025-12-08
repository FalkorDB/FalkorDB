/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "constraint.h"
#include "../query_ctx.h"
#include "../schema/schema.h"
#include "../graph/query_graph.h"
#include "../graph/entities/attribute_set.h"

#include <stdatomic.h>

// opaque structure representing a constraint
struct _MandatoryConstraint {
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
};

typedef struct _MandatoryConstraint* MandatoryConstraint;

static const char *_node_violation_err_msg =
	EMSG_MANDATORY_CONSTRAINT_VIOLATION_NODE;

static const char *_edge_violation_err_msg =
	EMSG_MANDATORY_CONSTRAINT_VIOLATION_EDGE;

// enforces mandatory constraint on given entity
bool Constraint_EnforceMandatory
(
	const Constraint c,    // constraint to enforce
	const GraphEntity *e,  // enforced entity
	char **err_msg         // report error message
) {
	MandatoryConstraint _c = (MandatoryConstraint)(c);

	for(uint8_t i = 0; i < _c->n_attr; i++) {
		AttributeID attr_id = _c->attrs[i];
		if (!GraphEntity_ContainsProperty (e, attr_id)) {
			// entity violates constraint
			if(err_msg != NULL) {
				// compose error message
				int res;
				UNUSED(res);

				GraphContext *gc = QueryCtx_GetGraphCtx();
				SchemaType st = (_c->et == GETYPE_NODE) ? SCHEMA_NODE : SCHEMA_EDGE;
				Schema *s = GraphContext_GetSchemaByID(gc, _c->schema_id, st);

				if(Constraint_GetEntityType(c) == GETYPE_NODE) {
					res = asprintf(err_msg, _node_violation_err_msg, Schema_GetName(s),
							_c->attr_names[i]);
				} else {
					res = asprintf(err_msg, _edge_violation_err_msg, Schema_GetName(s),
							_c->attr_names[i]);
				}
			}
			return false;
		}
	}

	// constraint holds for entity
	return true;
}

// create a new mandatory constraint
Constraint Constraint_MandatoryNew
(
	int schema_id,            // schema ID
	AttributeID *fields,      // enforced fields
	const char **attr_names,  // enforced attribute names
	uint8_t n_fields,         // number of fields
	GraphEntityType et        // entity type
) {
    MandatoryConstraint c = rm_malloc(sizeof(struct _MandatoryConstraint));

	// introduce constraint attributes
	c->attrs = rm_malloc(sizeof(AttributeID) * n_fields);
    memcpy(c->attrs, fields, sizeof(AttributeID) * n_fields);

	c->attr_names = rm_malloc(sizeof(char*) * n_fields);
    memcpy(c->attr_names, attr_names, sizeof(char*) * n_fields);

	// initialize constraint
	c->t               = CT_MANDATORY;
	c->et              = et;
	c->status          = CT_PENDING;
	c->n_attr          = n_fields;
	c->enforce         = Constraint_EnforceMandatory;
	c->set_pdata       = NULL;
	c->get_pdata       = NULL;
	c->schema_id       = schema_id;
	c->pending_changes = ATOMIC_VAR_INIT(0);

	return (Constraint)c;
}

