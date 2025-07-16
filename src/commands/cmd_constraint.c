/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "util/strutil.h"
#include "../query_ctx.h"
#include "../index/indexer.h"
#include "../graph/graph_hub.h"
#include "../undo_log/undo_log.h"
#include "../graph/graphcontext.h"
#include "constraint/constraint.h"

#define PROPERTY_NAME_PATTERN "[a-zA-Z_][a-zA-Z0-9_$]*"

// constraint operation
typedef enum {
	CT_CREATE,  // create constraint
	CT_DROP     // drop constraint
} ConstraintOp;

static inline int _cmp_Attribute_ID
(
	const void *a,
	const void *b
) {
	const Attribute_ID *_a = a;
	const Attribute_ID *_b = b;
	return *_a - *_b;
}

// parse command arguments
static int Constraint_Parse
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc,
	RedisModuleString **graph_name,
	ConstraintOp *op,
	ConstraintType *ct,
	GraphEntityType *type,
	const char **label,
	uint8_t *prop_count,
	RedisModuleString ***props
) {

	//--------------------------------------------------------------------------
	// get constraint operation CREATE/DROP
	//--------------------------------------------------------------------------

	const char *token = RedisModule_StringPtrLen(*argv++, NULL);
	if(strcasecmp(token, "CREATE") == 0) {
		*op = CT_CREATE;
	} else if(strcasecmp(token, "DROP") == 0) {
		*op = CT_DROP;
	} else {
		RedisModule_ReplyWithError(ctx, "Invalid constraint operation");
		return REDISMODULE_ERR;
	}

	//--------------------------------------------------------------------------
	// get graph name
	//--------------------------------------------------------------------------

	*graph_name = *argv++;

	//--------------------------------------------------------------------------
	// get constraint type UNIQUE/MANDATORY
	//--------------------------------------------------------------------------

	token = RedisModule_StringPtrLen(*argv++, NULL);
	if(strcasecmp(token, "UNIQUE") == 0) {
		*ct = CT_UNIQUE;
	} else if(strcasecmp(token, "MANDATORY") == 0) {
		*ct = CT_MANDATORY;
	} else {
		RedisModule_ReplyWithError(ctx, "Invalid constraint type");
		return REDISMODULE_ERR;
	}

	//--------------------------------------------------------------------------
	// extract entity type NODE/EDGE
	//--------------------------------------------------------------------------

	token = RedisModule_StringPtrLen(*argv++, NULL);
	if(strcasecmp(token, "NODE") == 0) {
		*type = GETYPE_NODE;
	} else if(strcasecmp(token, "RELATIONSHIP") == 0) {
		*type = GETYPE_EDGE;
	} else {
		RedisModule_ReplyWithError(ctx, "Invalid constraint entity type");
		return REDISMODULE_ERR;
	}

	//--------------------------------------------------------------------------
	// extract label/relationship-type
	//--------------------------------------------------------------------------

	*label = RedisModule_StringPtrLen(*argv++, NULL);
	if(str_MatchRegex(PROPERTY_NAME_PATTERN, *label) == false) {
		RedisModule_ReplyWithErrorFormat(ctx, "Label name %s is invalid", *label);
		return REDISMODULE_ERR;
	}

	//--------------------------------------------------------------------------
	// extract properties
	//--------------------------------------------------------------------------

	token = RedisModule_StringPtrLen(*argv++, NULL);
	long long _prop_count;
	if(strcasecmp(token, "PROPERTIES") == 0) {
		if(RedisModule_StringToLongLong(*argv++, &_prop_count) != REDISMODULE_OK
				|| _prop_count < 1 || _prop_count > 255) {
			RedisModule_ReplyWithError(ctx, "Number of properties must be an integer between 1 and 255");
			return REDISMODULE_ERR;
		}
	} else {
		RedisModule_ReplyWithError(ctx, "missing PROPERTIES argument");
		return REDISMODULE_ERR;
	}

	*prop_count = (uint8_t)(_prop_count);

	// expecting last property to be the last command argument
	if(argc - 7 != *prop_count) {
		RedisModule_ReplyWithError(ctx, "Number of properties doesn't match property count");
		return REDISMODULE_ERR;
	}

	*props = argv;

	return REDISMODULE_OK;
}

// GRAPH.CONSTRAIN <key> DROP UNIQUE/MANDATORY [NODE label / RELATIONSHIP type] PROPERTIES prop_count prop0, prop1...
static bool _Constraint_Drop
(
	RedisModuleCtx *ctx,    // redis module context
	RedisModuleString *key, // graph key to operate on
	ConstraintType ct,      // constraint type
	GraphEntityType et,     // entity type
	const char *lbl,        // label / rel-type
	uint8_t n,              // properties count
	const char **props      // properties
) {
	bool res = true;  // optimistic
	Attribute_ID attrs[n];

	//--------------------------------------------------------------------------
	// try to get graph
	//--------------------------------------------------------------------------

	GraphContext *gc = GraphContext_Retrieve(ctx, key, false, false);
	if(!gc) {
		// graph doesn't exists
		return false;
	}

	//--------------------------------------------------------------------------
	// try to get schema
	//--------------------------------------------------------------------------

	// determine schema type
	SchemaType st = (et == GETYPE_NODE) ? SCHEMA_NODE : SCHEMA_EDGE;

	Schema *s = GraphContext_GetSchema(gc, lbl, st);
	if(s == NULL) {
		res = false;
		goto cleanup;
	}

	//--------------------------------------------------------------------------
	// try to get attribute IDs
	//--------------------------------------------------------------------------

	for(uint8_t i = 0; i < n; i++) {
		const char *prop = props[i];

		// try to get property ID
		Attribute_ID id = GraphContext_GetAttributeID(gc, prop);

		if(id == ATTRIBUTE_ID_NONE) {
			// attribute missing
			res = false;
			goto cleanup;
		}

		attrs[i] = id;
	}

	//--------------------------------------------------------------------------
	// try to get constraint
	//--------------------------------------------------------------------------

	Constraint c = Schema_GetConstraint(s, ct, attrs, n);
	if(c == NULL) {
		res = false;
		goto cleanup;
	}

	//--------------------------------------------------------------------------
	// remove constraint
	//--------------------------------------------------------------------------

	// acquire graph write lock
	Graph_AcquireWriteLock(gc->g);

	Schema_RemoveConstraint(s, c);

	// release graph R/W lock
	Graph_ReleaseLock(gc->g);

	// TODO: consider disallowing droping a pending constraint
	// asynchronously delete constraint
	Indexer_DropConstraint(c, gc);

cleanup:
	if(res == false) {
		RedisModule_ReplyWithError(ctx, "Unable to drop constraint, no such constraint.");
	}

	// decrease graph reference count
	GraphContext_DecreaseRefCount(gc);

	return res;
}

// GRAPH.CONSTRAIN <key> CREATE UNIQUE/MANDATORY [NODE label / RELATIONSHIP type] PROPERTIES prop_count prop0, prop1...
static bool _Constraint_Create
(
	RedisModuleCtx *ctx,    // redis module context
	RedisModuleString *key, // graph key to operate on
	ConstraintType ct,      // constraint type
	GraphEntityType et,     // entity type
	const char *lbl,        // label / rel-type
	uint8_t n,              // properties count
	const char **props      // properties
) {
	bool res = true;
	const char *error_msg = "Constraint creation failed";

	// get or create graph
	GraphContext *gc = GraphContext_Retrieve(ctx, key, false, true);
	if(gc == NULL) {
		return false;	
	}

	// set graph context in query context TLS
	// this is required in case the undo-log needs to be applied
	// TODO: find a better way
	QueryCtx_SetGraphCtx(gc);

	Graph *g = GraphContext_GetGraph(gc);

	// acquire graph write lock
	Graph_AcquireWriteLock(gc->g);

	//--------------------------------------------------------------------------
	// convert attribute name to attribute ID
	//--------------------------------------------------------------------------

	Attribute_ID attr_ids[n];
	for(uint i = 0; i < n; i++) {
		attr_ids[i] = FindOrAddAttribute(gc, props[i], true);
	}

	//--------------------------------------------------------------------------
	// check for duplicates
	//--------------------------------------------------------------------------

	// sort the properties for an easy comparison later
	bool dups = false;
	qsort(attr_ids, n, sizeof(Attribute_ID), _cmp_Attribute_ID);
	for(uint i = 0; i < n - 1; i++) {
		if(attr_ids[i] == attr_ids[i+1]) {
			dups = true;
			break;
		}
	}

	// duplicates found, fail operation
	if(dups) {
		error_msg = "Properties cannot contain duplicates";
		res = false;
		goto cleanup;
	}

	// re-construct attribute IDs array
	// must be aligned with attribute names array
	for(uint i = 0; i < n; i++) {
		// get attribute id for attribute name
		Attribute_ID attr_id = attr_ids[i];

		// update props to hold graph context's attribute name
		props[i] = GraphContext_GetAttributeString(gc, attr_id);
	}

	//--------------------------------------------------------------------------
	// make sure schema exists
	//--------------------------------------------------------------------------

	SchemaType st = (et == GETYPE_NODE) ? SCHEMA_NODE : SCHEMA_EDGE;
	Schema *s = GraphContext_GetSchema(gc, lbl, st);
	if(s == NULL) {
		s = AddSchema(gc, lbl, st, true);
	}
	int s_id = Schema_GetID(s);

	//--------------------------------------------------------------------------
	// check if constraint already exists
	//--------------------------------------------------------------------------

	Constraint c = Schema_GetConstraint(s, ct, attr_ids, n);

	if(c != NULL) {
		// constraint already exists
		if(Constraint_GetStatus(c) != CT_FAILED) {
			// constraint is either operational or being constructed
			res = false;
			error_msg = "Constraint already exists";
			goto cleanup;
		} else {
			// previous constraint creation had failed
			// remove constrain from schema
			Schema_RemoveConstraint(s, c);

			// free failed constraint
			Constraint_Free(&c);
		}
	}
	
	//--------------------------------------------------------------------------
	// create constraint
	//--------------------------------------------------------------------------

	c = Constraint_New((struct GraphContext *)gc, ct, s_id, attr_ids, props, n,
			et, &error_msg);

	// failed to add constraint
	if(c == NULL) {
		res = false;
		goto cleanup;
	}

	// add constraint to schema
	Schema_AddConstraint(s, c);

cleanup:

	// operation failed perform clean up
	if(res == false) {
		QueryCtx_Rollback();
	}

	// release graph R/W lock
	Graph_ReleaseLock(g);

	// constraint already exists
	if(res == false) { 
		// TODO: give additional information to caller
		RedisModule_ReplyWithError(ctx, error_msg);
	} else {
		// constraint creation succeeded, enforce constraint
		Constraint_Enforce(c, (struct GraphContext*)gc);
	}

	QueryCtx_Free();

	// decrease graph reference count
	GraphContext_DecreaseRefCount(gc);

	return res;
}

// command handler for GRAPH.CONSTRAINT command
// GRAPH.CONSTRAINT CREATE <key> UNIQUE/MANDATORY [NODE label / RELATIONSHIP type] PROPERTIES prop_count prop0, prop1...
// GRAPH.CONSTRAINT DROP <key> UNIQUE/MANDATORY [NODE label / RELATIONSHIP type] PROPERTIES prop_count prop0, prop1...
int Graph_Constraint
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	if(argc < 8) {
		return RedisModule_WrongArity(ctx);
	}

	ConstraintOp op;
	ConstraintType ct;
	const char *label;
	uint8_t prop_count;
	RedisModuleString **props;
	GraphEntityType entity_type;
	RedisModuleString *key_name;

	//--------------------------------------------------------------------------
	// parse command arguments
	//--------------------------------------------------------------------------

	int res = Constraint_Parse(ctx, argv+1, argc-1, &key_name, &op, &ct,
			&entity_type, &label, &prop_count, &props);

	// command parsing error, abort
	if(res != REDISMODULE_OK) {
		return REDISMODULE_ERR;
	}

	// extract constraint properties and validate property name
	const char *props_cstr[prop_count];
	for(uint8_t i = 0; i < prop_count; i++) {
		props_cstr[i] = RedisModule_StringPtrLen(props[i], NULL);
		if(str_MatchRegex(PROPERTY_NAME_PATTERN, props_cstr[i]) == false) {
			RedisModule_ReplyWithErrorFormat(ctx, "Property name %s is invalid", props_cstr[i]);
			return REDISMODULE_ERR;
		}
	}

	bool success = false;

	if(op == CT_CREATE) {
		// try to create constraint
		success = _Constraint_Create(ctx, key_name, ct, entity_type, label,
				prop_count, props_cstr);
	} else {
		// CT_DROP
		success = _Constraint_Drop(ctx, key_name, ct, entity_type, label,
				prop_count, props_cstr);
	}

	if(success == true) {
		RedisModule_ReplyWithSimpleString(ctx, op == CT_CREATE ? "PENDING" : "OK");
		RedisModule_ReplicateVerbatim(ctx);
		return REDISMODULE_OK;
	}

	return REDISMODULE_ERR;
}

