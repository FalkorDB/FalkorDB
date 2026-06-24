/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "util/strutil.h"
#include "../query_ctx.h"
#include "../index/indexer.h"
#include "../graph/graph_hub.h"
#include "../graph/graphcontext.h"
#include "constraint/constraint.h"
#include "../util/thpool/pool.h"

#define PROPERTY_NAME_PATTERN "[a-zA-Z_][a-zA-Z0-9_$]*"

// constraint operation
typedef enum {
	CT_CREATE,  // create constraint
	CT_DROP     // drop constraint
} ConstraintOp;

// GRAPH.CONSTRAINT command context
typedef struct {
	RedisModuleBlockedClient *bc;           // blocked client
	RedisModuleString        *graph_id;     // graph name
	ConstraintOp              op;           // CREATE or DROP
	ConstraintType            ct;           // UNIQUE or MANDATORY
	GraphEntityType           entity_type;  // NODE or RELATIONSHIP
	char                     *label;        // label / rel-type (owned copy)
	uint8_t                   prop_count;   // number of properties
	char                    **props;        // prop_count owned property name copies
} GraphConstraintCtx;

static GraphConstraintCtx *GraphConstraintCtx_New
(
	ConstraintOp op,
	ConstraintType ct,
	const char *label,
	uint8_t prop_count,
	GraphEntityType entity_type
) {
	ASSERT (label != NULL) ;
	ASSERT (prop_count > 0) ;

	GraphConstraintCtx *ctx = rm_calloc (1, sizeof (GraphConstraintCtx)) ;

	ctx->op          = op ;
	ctx->ct          = ct ;
	ctx->label       = rm_strdup (label) ;
	ctx->prop_count  = prop_count ;
	ctx->entity_type = entity_type ;

	return ctx ;
}

static void GraphConstraintCtx_Free
(
	RedisModuleCtx *rm_ctx,
	GraphConstraintCtx **ctx
) {
	GraphConstraintCtx *_ctx = *ctx ;

	RedisModule_FreeString (rm_ctx, _ctx->graph_id) ;

	for (uint8_t i = 0 ; i < _ctx->prop_count ; i++) {
		rm_free (_ctx->props [i]) ;
	}

	rm_free (_ctx->props) ;
	rm_free (_ctx->label) ;
	rm_free (_ctx) ;

	*ctx = NULL ;
}

static inline int _cmp_AttributeID
(
	const void *a,
	const void *b
) {
	const AttributeID *_a = a;
	const AttributeID *_b = b;
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
	bool res = true ;  // optimistic
	AttributeID attrs [n] ;
	RedisModuleString *prop_strs [n] ;

	//--------------------------------------------------------------------------
	// try to get graph
	//--------------------------------------------------------------------------

	GraphContext *gc = NULL ;
	if (GraphContext_Retrieve (ctx, key, false, false, true, &gc)
			!= GraphRetrieve_RETRIEVED) {
		// graph doesn't exists
		return false ;
	}

	// acquire graph write lock
	RedisModule_ThreadSafeContextLock (ctx) ;
	GraphContext_AcquireWriteLock (gc) ;

	//--------------------------------------------------------------------------
	// try to get schema
	//--------------------------------------------------------------------------

	// determine schema type
	SchemaType st = (et == GETYPE_NODE) ? SCHEMA_NODE : SCHEMA_EDGE ;

	Schema *s = GraphContext_GetSchema (gc, lbl, st) ;
	if (s == NULL) {
		res = false ;
		goto cleanup ;
	}

	//--------------------------------------------------------------------------
	// try to get attribute IDs
	//--------------------------------------------------------------------------

	for (uint8_t i = 0 ; i < n ; i++) {
		const char *prop = props [i] ;

		// try to get property ID
		AttributeID id = GraphContext_GetAttributeID (gc, prop) ;

		if (id == ATTRIBUTE_ID_NONE) {
			// attribute missing
			res = false ;
			goto cleanup ;
		}

		attrs [i] = id ;
	}

	//--------------------------------------------------------------------------
	// try to get constraint
	//--------------------------------------------------------------------------

	Constraint c = Schema_GetConstraint (s, ct, attrs, n) ;
	if (c == NULL) {
		res = false ;
		goto cleanup ;
	}

	//--------------------------------------------------------------------------
	// remove constraint
	//--------------------------------------------------------------------------

	Schema_RemoveConstraint (s, c) ;

	// TODO: consider disallowing droping a pending constraint
	// asynchronously delete constraint
	Indexer_DropConstraint (c, gc) ;

	//--------------------------------------------------------------------------
	// replicate DROP to replicas and persistence layer
	//--------------------------------------------------------------------------

	const char *graph_name = GraphContext_GetName (gc) ;
	const char *c_type = (ct == CT_UNIQUE)   ? "UNIQUE" : "MANDATORY" ;
	const char *et_str = (et == GETYPE_NODE) ? "NODE"   : "RELATIONSHIP" ;

	// resolve attribute IDs to names
	for (uint8_t i = 0 ; i < n ; i++) {
		const char *attr_name = GraphContext_GetAttributeName (gc, attrs [i]) ;
		prop_strs [i] = RedisModule_CreateString (ctx, attr_name,
				strlen (attr_name)) ;
	}

	RedisModule_Replicate (ctx, "GRAPH.CONSTRAINT", "cccccclv",
			"DROP", graph_name, c_type, et_str, lbl,
			"PROPERTIES", (long long)n, prop_strs, (size_t)n) ;

	for (uint8_t i = 0 ; i < n ; i++) {
		RedisModule_FreeString (ctx, prop_strs [i]) ;
	}

cleanup:
	// release graph R/W lock
	GraphContext_ReleaseLock (gc) ;
	RedisModule_ThreadSafeContextUnlock (ctx) ;

	if (res == false) {
		RedisModule_ReplyWithError (ctx,
				"Unable to drop constraint, no such constraint.") ;
	}

	// decrease graph reference count
	GraphContext_DecreaseRefCount (gc) ;

	return res ;
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
	GraphContext *gc = NULL ;
	if (GraphContext_Retrieve (ctx, key, false, true, true, &gc)
			!= GraphRetrieve_RETRIEVED) {
		return false ;
	}

	// set graph context in query context TLS
	// this is required in case the undo-log needs to be applied
	// TODO: find a better way
	QueryCtx_SetGraphCtx (gc) ;

	// acquire graph write lock
	RedisModule_ThreadSafeContextLock (ctx) ;
	GraphContext_AcquireWriteLock (gc) ;

	//--------------------------------------------------------------------------
	// convert attribute name to attribute ID
	//--------------------------------------------------------------------------

	AttributeID attr_ids [n] ;
	for (uint i = 0 ; i < n ; i++) {
		attr_ids [i] = GraphHub_FindOrAddAttribute (gc, props [i], true) ;
	}

	//--------------------------------------------------------------------------
	// check for duplicates
	//--------------------------------------------------------------------------

	// sort the properties for an easy comparison later
	bool dups = false ;
	qsort (attr_ids, n, sizeof (AttributeID), _cmp_AttributeID) ;
	for (uint i = 0 ; i < n - 1 ; i++) {
		if (attr_ids [i] == attr_ids [i+1]) {
			dups = true ;
			break ;
		}
	}

	// duplicates found, fail operation
	if (dups) {
		error_msg = "Properties cannot contain duplicates" ;
		res = false ;
		goto cleanup ;
	}

	// re-construct attribute IDs array
	// must be aligned with attribute names array
	for (uint i = 0 ; i < n ; i++) {
		// get attribute id for attribute name
		AttributeID attr_id = attr_ids [i] ;

		// update props to hold graph context's attribute name
		props [i] = GraphContext_GetAttributeName (gc, attr_id) ;
	}

	//--------------------------------------------------------------------------
	// make sure schema exists
	//--------------------------------------------------------------------------

	SchemaType st = (et == GETYPE_NODE) ? SCHEMA_NODE : SCHEMA_EDGE;
	Schema *s = GraphContext_GetSchema (gc, lbl, st) ;
	if (s == NULL) {
		s = GraphHub_AddSchema (gc, lbl, st, true) ;
	}
	int s_id = Schema_GetID (s) ;

	//--------------------------------------------------------------------------
	// check if constraint already exists
	//--------------------------------------------------------------------------

	Constraint c = Schema_GetConstraint (s, ct, attr_ids, n) ;

	if (c != NULL) {
		// constraint already exists
		if (Constraint_GetStatus (c) != CT_FAILED) {
			// constraint is either operational or being constructed
			res = false ;
			error_msg = "Constraint already exists" ;
			goto cleanup ;
		} else {
			// previous constraint creation had failed
			// remove constrain from schema
			Schema_RemoveConstraint (s, c) ;

			// free failed constraint
			Constraint_Free (&c) ;
		}
	}
	
	//--------------------------------------------------------------------------
	// create constraint
	//--------------------------------------------------------------------------

	c = Constraint_New ((struct GraphContext *)gc, ct, s_id, attr_ids, props, n,
			et, &error_msg) ;

	// failed to add constraint
	if (c == NULL) {
		res = false ;
		goto cleanup ;
	}

	// add constraint to schema
	Schema_AddConstraint (s, c) ;

	// replication requires the Redis global lock
	Constraint_Replicate (ctx, c, gc) ;

cleanup:

	// operation failed perform clean up
	if (res == false) {
		QueryCtx_Rollback () ;
	}

	// release graph R/W lock
	GraphContext_ReleaseLock (gc) ;
	RedisModule_ThreadSafeContextUnlock (ctx) ;

	// constraint already exists
	if (res == false) {
		// TODO: give additional information to caller
		RedisModule_ReplyWithError (ctx, error_msg) ;
	} else {
		// constraint creation succeeded, enforce constraint
		Constraint_Enforce (c, (struct GraphContext*)gc) ;
	}

	QueryCtx_Free () ;

	// decrease graph reference count
	GraphContext_DecreaseRefCount (gc) ;

	return res ;
}

// GRAPH.CONSTRAINT internal command handler
// executed on a worker thread to avoid blocking the main thread
static void _Graph_Constraint
(
	void *_ctx  // command context
) {
	ASSERT(_ctx != NULL);

	GraphConstraintCtx       *ctx    = (GraphConstraintCtx *)_ctx ;
	RedisModuleBlockedClient *bc     = ctx->bc ;
	RedisModuleCtx           *rm_ctx = RedisModule_GetThreadSafeContext (bc) ;

	// build working props array — _Constraint_Create may overwrite elements
	// with GraphContext-internal pointers, so keep ctx->props for cleanup
	const char *props [ctx->prop_count] ;
	for (uint8_t i = 0; i < ctx->prop_count; i++) {
		props[i] = ctx->props [i] ;
	}

	bool success = false ;

	if(ctx->op == CT_CREATE) {
		success = _Constraint_Create (rm_ctx, ctx->graph_id, ctx->ct,
				ctx->entity_type, ctx->label, ctx->prop_count, props) ;
	} else {
		success = _Constraint_Drop (rm_ctx, ctx->graph_id, ctx->ct,
				ctx->entity_type, ctx->label, ctx->prop_count, props) ;
	}

	if (success) {
		RedisModule_ReplyWithSimpleString (rm_ctx,
				ctx->op == CT_CREATE ? "PENDING" : "OK") ;
	}

	// cleanup
	GraphConstraintCtx_Free (rm_ctx, &ctx) ;

	// unblock client
	RedisModule_UnblockClient (bc, NULL) ;

	// free thread-safe context
	RedisModule_FreeThreadSafeContext (rm_ctx) ;
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
	if (argc < 8) {
		return RedisModule_WrongArity (ctx) ;
	}

	ConstraintOp op ;
	ConstraintType ct ;
	const char *label ;
	GraphEntityType entity_type ;

	uint8_t prop_count          = 0 ;
	RedisModuleString **props   = NULL ;
	RedisModuleString *key_name = NULL ;

	//--------------------------------------------------------------------------
	// parse command arguments
	//--------------------------------------------------------------------------

	int res = Constraint_Parse (ctx, argv+1, argc-1, &key_name, &op, &ct,
			&entity_type, &label, &prop_count, &props) ;

	// command parsing error, abort
	if (res != REDISMODULE_OK) {
		return REDISMODULE_ERR ;
	}

	//--------------------------------------------------------------------------
	// extract constraint properties and validate property names
	//--------------------------------------------------------------------------

	const char *props_cstr [prop_count] ;
	for (uint8_t i = 0 ; i < prop_count ; i++) {
		props_cstr [i] = RedisModule_StringPtrLen (props [i], NULL) ;
		if (str_MatchRegex (PROPERTY_NAME_PATTERN, props_cstr [i]) == false) {
			RedisModule_ReplyWithErrorFormat (ctx, "Property name %s is invalid",
					props_cstr [i]) ;
			return REDISMODULE_ERR ;
		}
	}

	// build context for worker thread
	GraphConstraintCtx *cmd_ctx =
		GraphConstraintCtx_New (op, ct, label, prop_count, entity_type) ;

	// copy property names — the worker may pass them to _Constraint_Create
	// which overwrites the array elements with GraphContext-internal pointers
	cmd_ctx->props = rm_malloc (prop_count * sizeof (char *)) ;
	for (uint8_t i = 0 ; i < prop_count ; i++) {
		cmd_ctx->props [i] = rm_strdup (props_cstr [i]) ;
	}

	// retain graph name for use on the worker thread
	RedisModule_RetainString (ctx, key_name) ;
	cmd_ctx->graph_id = key_name ;

	// block client and dispatch to thread pool
	cmd_ctx->bc = RedisModule_BlockClient (ctx, NULL, NULL, NULL, 0) ;

	ThreadPool_AddWork (_Graph_Constraint, cmd_ctx, true) ;

	return REDISMODULE_OK;
}

