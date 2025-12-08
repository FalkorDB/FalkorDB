/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "../resultset.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "resultset_formatters.h"
#include "../../datatypes/datatypes.h"

// forward declarations
static void _ResultSet_VerboseReplyWithMap(RedisModuleCtx *ctx, SIValue map);
static void _ResultSet_VerboseReplyWithPath(RedisModuleCtx *ctx, SIValue path);
static void _ResultSet_VerboseReplyWithPoint(RedisModuleCtx *ctx, SIValue point);
static void _ResultSet_VerboseReplyWithArray(RedisModuleCtx *ctx, SIValue array);
static void _ResultSet_VerboseReplyWithNode(RedisModuleCtx *ctx, GraphContext *gc, Node *n);
static void _ResultSet_VerboseReplyWithEdge(RedisModuleCtx *ctx, GraphContext *gc, Edge *e);
static void _ResultSet_VerboseReplyWithVector(RedisModuleCtx *ctx, SIValue vector);
static void _ResultSet_VerboseReplyAsString(RedisModuleCtx *ctx, SIValue v);

// this function has handling for all SIValue scalar types
// the current RESP protocol only has unique support for:
// strings, 8-byte integers, and NULL values
static void _ResultSet_VerboseReplyWithSIValue
(
	RedisModuleCtx *ctx,
	GraphContext *gc,
	const SIValue v
) {
	switch(SI_TYPE(v)) {
		case T_STRING:
		case T_INTERN_STRING:
			RedisModule_ReplyWithStringBuffer(ctx, v.stringval,
					strlen(v.stringval));
			break;

		case T_INT64:
			RedisModule_ReplyWithLongLong(ctx, v.longval);
			break;

		case T_DOUBLE:
			_ResultSet_ReplyWithRoundedDouble(ctx, v.doubleval);
			break;

		case T_BOOL:
			if(v.longval != 0) RedisModule_ReplyWithStringBuffer(ctx, "true", 4);
			else RedisModule_ReplyWithStringBuffer(ctx, "false", 5);
			break;

		case T_NULL:
			RedisModule_ReplyWithNull(ctx);
			break;

		case T_NODE:
			_ResultSet_VerboseReplyWithNode(ctx, gc, v.ptrval);
			break;

		case T_EDGE:
			_ResultSet_VerboseReplyWithEdge(ctx, gc, v.ptrval);
			break;

		case T_ARRAY:
			_ResultSet_VerboseReplyWithArray(ctx, v);
			break;

		case T_PATH:
			_ResultSet_VerboseReplyWithPath(ctx, v);
			break;

		case T_MAP:
			_ResultSet_VerboseReplyWithMap(ctx, v);
			break;

		case T_POINT:
			_ResultSet_VerboseReplyWithPoint(ctx, v);
			break;

		case T_VECTOR_F32:
			_ResultSet_VerboseReplyWithVector(ctx, v);
			break;

		case T_DATETIME:
			_ResultSet_VerboseReplyAsString(ctx, v);
			break;

		case T_DATE:
			_ResultSet_VerboseReplyAsString(ctx, v);
			break;

		case T_TIME:
			_ResultSet_VerboseReplyAsString(ctx, v);
			break;

		case T_DURATION:
			_ResultSet_VerboseReplyAsString(ctx, v);
			break;

		default:
			RedisModule_Assert("Unhandled value type" && false);
	}
}

static void _ResultSet_VerboseReplyWithProperties
(
	RedisModuleCtx *ctx,
	GraphContext *gc,
	const GraphEntity *e
) {
	const AttributeSet set = GraphEntity_GetAttributes (e) ;
	int prop_count = AttributeSet_Count (set) ;
	RedisModule_ReplyWithArray (ctx, prop_count) ;

	// iterate over all properties stored on entity
	for(int i = 0; i < prop_count; i ++) {
		RedisModule_ReplyWithArray (ctx, 2) ;

		SIValue value ;
		AttributeID attr_id ;
		AttributeSet_GetIdx (set, i, &attr_id, &value) ;

		// emit the actual string
		const char *prop_str = GraphContext_GetAttributeString (gc, attr_id) ;
		RedisModule_ReplyWithStringBuffer (ctx, prop_str, strlen (prop_str)) ;
		// emit the value
		_ResultSet_VerboseReplyWithSIValue (ctx, gc, value) ;
	}
}

static void _ResultSet_VerboseReplyWithNode(RedisModuleCtx *ctx, GraphContext *gc, Node *n) {
	/*  Verbose node reply format:
	 *  [
	 *      ["id", Node ID (integer)]
	 *      ["label", [label (NULL or string X N)]]
	 *      ["properties", [[name, value, value type] X N]
	 *  ]
	 */
	// 3 top-level entities in node reply
	RedisModule_ReplyWithArray(ctx, 3);

	// ["id", id (integer)]
	EntityID id = ENTITY_GET_ID(n);
	RedisModule_ReplyWithArray(ctx, 2);
	RedisModule_ReplyWithStringBuffer(ctx, "id", 2);
	RedisModule_ReplyWithLongLong(ctx, id);

	// ["labels", [label (string) X N]]
	RedisModule_ReplyWithArray(ctx, 2);
	RedisModule_ReplyWithStringBuffer(ctx, "labels", 6);

	uint lbls_count;
	NODE_GET_LABELS(gc->g, n, lbls_count);
	RedisModule_ReplyWithArray(ctx, lbls_count);
	for(int i = 0; i < lbls_count; i++) {
		Schema *s = GraphContext_GetSchemaByID(gc, labels[i], SCHEMA_NODE);
		const char *lbl_name = Schema_GetName(s);
		RedisModule_ReplyWithStringBuffer(ctx, lbl_name, strlen(lbl_name));
	}

	// [properties, [properties]]
	RedisModule_ReplyWithArray(ctx, 2);
	RedisModule_ReplyWithStringBuffer(ctx, "properties", 10);
	_ResultSet_VerboseReplyWithProperties(ctx, gc, (GraphEntity *)n);
}

static void _ResultSet_VerboseReplyWithEdge(RedisModuleCtx *ctx, GraphContext *gc, Edge *e) {
	/*  Edge reply format:
	 *  [
	 *      ["id", Edge ID (integer)]
	 *      ["type", relation type (string)]
	 *      ["src_node", source node ID (integer)]
	 *      ["dest_node", destination node ID (integer)]
	 *      ["properties", [[name, value, value type] X N]
	 *  ]
	 */
	// 5 top-level entities in edge reply
	RedisModule_ReplyWithArray(ctx, 5);

	// ["id", id (integer)]
	RedisModule_ReplyWithArray(ctx, 2);
	RedisModule_ReplyWithStringBuffer(ctx, "id", 2);
	RedisModule_ReplyWithLongLong(ctx, ENTITY_GET_ID(e));

	// ["type", type (string)]
	RedisModule_ReplyWithArray(ctx, 2);
	RedisModule_ReplyWithStringBuffer(ctx, "type", 4);
	// Retrieve relation type
	Schema *s = GraphContext_GetSchemaByID(gc, Edge_GetRelationID(e), SCHEMA_EDGE);
	const char *reltype = Schema_GetName(s);
	RedisModule_ReplyWithStringBuffer(ctx, reltype, strlen(reltype));

	// ["src_node", src_id (integer)]
	RedisModule_ReplyWithArray(ctx, 2);
	RedisModule_ReplyWithStringBuffer(ctx, "src_node", 8);
	RedisModule_ReplyWithLongLong(ctx, Edge_GetSrcNodeID(e));

	// ["dest_node", dest_id (integer)]
	RedisModule_ReplyWithArray(ctx, 2);
	RedisModule_ReplyWithStringBuffer(ctx, "dest_node", 9);
	RedisModule_ReplyWithLongLong(ctx, Edge_GetDestNodeID(e));

	// [properties, [properties]]
	RedisModule_ReplyWithArray(ctx, 2);
	RedisModule_ReplyWithStringBuffer(ctx, "properties", 10);
	_ResultSet_VerboseReplyWithProperties(ctx, gc, (GraphEntity *)e);
}

static void _ResultSet_VerboseReplyWithArray(RedisModuleCtx *ctx, SIValue array) {
	size_t bufferLen = 512;
	char *str = rm_calloc(bufferLen, sizeof(char));
	size_t bytesWrriten = 0;
	SIValue_ToString(array, &str, &bufferLen, &bytesWrriten);
	RedisModule_ReplyWithStringBuffer(ctx, str, bytesWrriten);
	rm_free(str);
}

static void _ResultSet_VerboseReplyWithPath(RedisModuleCtx *ctx, SIValue path) {
	SIValue path_array = SIPath_ToList(path);
	_ResultSet_VerboseReplyWithArray(ctx, path_array);
	SIValue_Free(path_array);
}

static void _ResultSet_VerboseReplyWithMap(RedisModuleCtx *ctx, SIValue map) {
	size_t bufferLen = 512;
	char *str = rm_calloc(bufferLen, sizeof(char));
	size_t bytesWrriten = 0;
	SIValue_ToString(map, &str, &bufferLen, &bytesWrriten);
	RedisModule_ReplyWithStringBuffer(ctx, str, bytesWrriten);
	rm_free(str);
}

static void _ResultSet_VerboseReplyWithPoint(RedisModuleCtx *ctx, SIValue point) {
	// point({latitude:56.7, longitude:12.78})
	char buffer[256];
	int bytes_written = sprintf(buffer, "point({latitude:%f, longitude:%f})",
			Point_lat(point), Point_lon(point));

	RedisModule_ReplyWithStringBuffer(ctx, buffer, bytes_written);
}

static void _ResultSet_VerboseReplyWithVector
(
	RedisModuleCtx *ctx,
	SIValue vector
) {
	size_t bufferLen = 512;
	char *str = rm_calloc(bufferLen, sizeof(char));
	size_t bytesWrriten = 0;
	SIValue_ToString(vector, &str, &bufferLen, &bytesWrriten);
	RedisModule_ReplyWithStringBuffer(ctx, str, bytesWrriten);
	rm_free(str);
}

static void _ResultSet_VerboseReplyAsString
(
	RedisModuleCtx *ctx,
	SIValue v
) {
	char buffer[128];
	char *bufPtr = buffer;
	size_t bufferLen = 128;
	size_t bytesWrriten = 0;
	SIValue_ToString(v, (char**)&bufPtr, &bufferLen, &bytesWrriten);

	RedisModule_ReplyWithStringBuffer(ctx, bufPtr, strlen(bufPtr));
}

void ResultSet_EmitVerboseRow
(
	ResultSet *set,
	SIValue **row
) {
	RedisModuleCtx *ctx = set->ctx;
	// Prepare return array sized to the number of RETURN entities
	RedisModule_ReplyWithArray(ctx, set->column_count);

	for(int i = 0; i < set->column_count; i++) {
		SIValue v = *row[i];
		_ResultSet_VerboseReplyWithSIValue(ctx, set->gc, v);
	}
}

// Emit the alias or descriptor for each column in the header.
void ResultSet_ReplyWithVerboseHeader
(
	ResultSet *set
) {
	RedisModuleCtx *ctx = set->ctx;
	if(set->column_count > 0) {
		// prepare a response containing a header, records, and statistics
		RedisModule_ReplyWithArray(ctx, 3);
	} else {
		// prepare a response containing only statistics
		RedisModule_ReplyWithArray(ctx, 1);
		return;
	}
	RedisModule_ReplyWithArray(ctx, set->column_count);
	for(uint i = 0; i < set->column_count; i++) {
		// Emit the identifier string associated with the column
		RedisModule_ReplyWithStringBuffer(ctx, set->columns[i], strlen(set->columns[i]));
	}
}

void ResultSet_EmitVerboseStats
(
	ResultSet *set
) {
	RedisModuleCtx *ctx = set->ctx;
	ResultSetStatistics *stats = &set->stats;
	int buflen;
	char buff[512] = {0};
	size_t resultset_size = 2; // execution time, cached

	// compute required space for resultset statistics
	if(stats->index_creation)            resultset_size++;
	if(stats->index_deletion)            resultset_size++;
	if(stats->constraint_creation)       resultset_size++;
	if(stats->constraint_deletion)       resultset_size++;
	if(stats->labels_added          > 0) resultset_size++;
	if(stats->nodes_created         > 0) resultset_size++;
	if(stats->nodes_deleted         > 0) resultset_size++;
	if(stats->labels_removed        > 0) resultset_size++;
	if(stats->properties_set        > 0) resultset_size++;
	if(stats->properties_removed    > 0) resultset_size++;
	if(stats->relationships_deleted > 0) resultset_size++;
	if(stats->relationships_created > 0) resultset_size++;

	RedisModule_ReplyWithArray(ctx, resultset_size);

	if(stats->labels_added > 0) {
		buflen = sprintf(buff, "Labels added: %d", stats->labels_added);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->labels_removed > 0) {
		buflen = sprintf(buff, "Labels removed: %d", stats->labels_removed);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->nodes_created > 0) {
		buflen = sprintf(buff, "Nodes created: %d", stats->nodes_created);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->properties_set > 0) {
		buflen = sprintf(buff, "Properties set: %d", stats->properties_set);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->properties_removed > 0) {
		buflen = sprintf(buff, "Properties removed: %d", stats->properties_removed);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->relationships_created > 0) {
		buflen = sprintf(buff, "Relationships created: %d", stats->relationships_created);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->nodes_deleted > 0) {
		buflen = sprintf(buff, "Nodes deleted: %d", stats->nodes_deleted);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->relationships_deleted > 0) {
		buflen = sprintf(buff, "Relationships deleted: %d", stats->relationships_deleted);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->index_creation) {
		buflen = sprintf(buff, "Indices created: %d", stats->indices_created);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->index_deletion) {
		buflen = sprintf(buff, "Indices deleted: %d", stats->indices_deleted);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->constraint_creation) {
		buflen = sprintf(buff, "Constraints created: %d", stats->constraints_created);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->constraint_deletion) {
		buflen = sprintf(buff, "Constraints deleted: %d", stats->constraints_deleted);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	buflen = sprintf(buff, "Cached execution: %d", stats->cached ? 1 : 0);
	RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);

	// emit query execution time
	double t = QueryCtx_GetRuntime();
	buflen = sprintf(buff, "Query internal execution time: %.6f milliseconds", t);
	RedisModule_ReplyWithStringBuffer(ctx, buff, buflen);
}
