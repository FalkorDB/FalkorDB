/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../resultset.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "resultset_formatters.h"
#include "../../datatypes/datatypes.h"

// forward declarations
static void _ResultSet_CompactReplyWithNode(RedisModuleCtx *ctx, GraphContext *gc, Node *n);
static void _ResultSet_CompactReplyWithEdge(RedisModuleCtx *ctx, GraphContext *gc, Edge *e);
static void _ResultSet_CompactReplyWithSIArray(RedisModuleCtx *ctx, GraphContext *gc, SIValue array);
static void _ResultSet_CompactReplyWithPath(RedisModuleCtx *ctx, GraphContext *gc, SIValue path);
static void _ResultSet_CompactReplyWithMap(RedisModuleCtx *ctx, GraphContext *gc, SIValue v);
static void _ResultSet_CompactReplyWithPoint(RedisModuleCtx *ctx, GraphContext *gc, SIValue v);
static void _ResultSet_CompactReplyWithVector32F(RedisModuleCtx *ctx, SIValue vec);

static inline ValueType _mapValueType
(
	SIType t
) {
	switch(t) {
	case T_NULL:
		return VALUE_NULL;

	case T_STRING:
	case T_INTERN_STRING:
		return VALUE_STRING;

	case T_INT64:
		return VALUE_INTEGER;

	case T_BOOL:
		return VALUE_BOOLEAN;

	case T_DOUBLE:
		return VALUE_DOUBLE;

	case T_ARRAY:
		return VALUE_ARRAY;

	case T_VECTOR_F32:
		return VALUE_VECTORF32;

	case T_NODE:
		return VALUE_NODE;

	case T_EDGE:
		return VALUE_EDGE;

	case T_PATH:
		return VALUE_PATH;

	case T_MAP:
		return VALUE_MAP;

	case T_POINT:
		return VALUE_POINT;

	case T_DATETIME:
		return VALUE_DATETIME;

	case T_DATE:
		return VALUE_DATE;

	case T_TIME:
		return VALUE_TIME;

	case T_DURATION:
		return VALUE_DURATION;

	default:
		return VALUE_UNKNOWN;
	}
}

static inline void _ResultSet_ReplyWithValueType
(
	RedisModuleCtx *ctx,
	SIType t
) {
	RedisModule_ReplyWithLongLong(ctx, _mapValueType(t));
}

// emit value to reply stream
static void _ResultSet_CompactReplyWithSIValue
(
	RedisModuleCtx *ctx,  // redis context
	GraphContext *gc,     // graph context
	const SIValue v       // value to emit
) {
	// Emit the value type, then the actual value (to facilitate client-side parsing)
	_ResultSet_ReplyWithValueType(ctx, SI_TYPE(v));

	switch(SI_TYPE(v)) {
	case T_STRING:
	case T_INTERN_STRING:
		RedisModule_ReplyWithStringBuffer(ctx, v.stringval, strlen(v.stringval));
		return;

	case T_INT64:
		RedisModule_ReplyWithLongLong(ctx, v.longval);
		return;

	case T_DOUBLE:
		_ResultSet_ReplyWithRoundedDouble(ctx, v.doubleval);
		return;

	case T_BOOL:
		if(v.longval != 0) RedisModule_ReplyWithStringBuffer(ctx, "true", 4);
		else RedisModule_ReplyWithStringBuffer(ctx, "false", 5);
		return;

	case T_TIME:
	case T_DATE:
	case T_DATETIME:
	case T_DURATION:
		RedisModule_ReplyWithLongLong(ctx, v.datetimeval);
		return;

	case T_ARRAY:
		_ResultSet_CompactReplyWithSIArray(ctx, gc, v);
		break;

	case T_VECTOR_F32:
		_ResultSet_CompactReplyWithVector32F(ctx, v);
		break;

	case T_NULL:
		RedisModule_ReplyWithNull(ctx);
		return;

	case T_NODE:
		_ResultSet_CompactReplyWithNode(ctx, gc, v.ptrval);
		return;

	case T_EDGE:
		_ResultSet_CompactReplyWithEdge(ctx, gc, v.ptrval);
		return;

	case T_PATH:
		_ResultSet_CompactReplyWithPath(ctx, gc, v);
		return;

	case T_MAP:
		_ResultSet_CompactReplyWithMap(ctx, gc, v);
		return;

	case T_POINT:
		_ResultSet_CompactReplyWithPoint(ctx, gc, v);
		return;

	default:
		RedisModule_Assert("Unhandled value type" && false);
		break;
	}
}

static void _ResultSet_CompactReplyWithProperties
(
	RedisModuleCtx *ctx,
	GraphContext *gc,
	const GraphEntity *e
) {
	const AttributeSet set = GraphEntity_GetAttributes(e);
	int prop_count = AttributeSet_Count(set);
	RedisModule_ReplyWithArray(ctx, prop_count);

	// iterate over all properties stored on entity
	for (int i = 0; i < prop_count; i++) {
		// compact replies include the value's type; verbose replies do not
		RedisModule_ReplyWithArray (ctx, 3) ;
		SIValue value ;
		AttributeID attr_id;
		AttributeSet_GetIdx (set, i, &attr_id, &value) ;

		// emit the attribute id
		RedisModule_ReplyWithLongLong (ctx, attr_id) ;

		// emit the value
		_ResultSet_CompactReplyWithSIValue (ctx, gc, value) ;
	}
}

static void _ResultSet_CompactReplyWithNode
(
	RedisModuleCtx *ctx,
	GraphContext *gc,
	Node *n
) {
	/*  Compact node reply format:
	 *  [
	 *      Node ID (integer),
	        [label string index (integer) X N],
	 *      [[name, value, value type] X N]
	 *  ]
	 */
	// 3 top-level entities in node reply
	RedisModule_ReplyWithArray(ctx, 3);

	// id (integer)
	EntityID id = ENTITY_GET_ID(n);
	RedisModule_ReplyWithLongLong(ctx, id);

	// [label string index X N]
	// Retrieve node labels
	uint lbls_count;
	NODE_GET_LABELS(gc->g, n, lbls_count);
	RedisModule_ReplyWithArray(ctx, lbls_count);
	for(int i = 0; i < lbls_count; i++) {
		RedisModule_ReplyWithLongLong(ctx, labels[i]);
	}

	// [properties]
	_ResultSet_CompactReplyWithProperties(ctx, gc, (GraphEntity *)n);
}

static void _ResultSet_CompactReplyWithEdge
(
	RedisModuleCtx *ctx,
	GraphContext *gc,
	Edge *e
) {
	/*  Compact edge reply format:
	 *  [
	 *      Edge ID (integer),
	        reltype string index (integer),
	        src node ID (integer),
	        dest node ID (integer),
	 *      [[name, value, value type] X N]
	 *  ]
	 */
	// 5 top-level entities in edge reply
	RedisModule_ReplyWithArray(ctx, 5);

	// id (integer)
	EntityID id = ENTITY_GET_ID(e);
	RedisModule_ReplyWithLongLong(ctx, id);

	// reltype string index, retrieve reltype.
	int reltype_id = Edge_GetRelationID(e);
	ASSERT(reltype_id != GRAPH_NO_RELATION);
	RedisModule_ReplyWithLongLong(ctx, reltype_id);

	// src node ID
	RedisModule_ReplyWithLongLong(ctx, Edge_GetSrcNodeID(e));

	// dest node ID
	RedisModule_ReplyWithLongLong(ctx, Edge_GetDestNodeID(e));

	// [properties]
	_ResultSet_CompactReplyWithProperties(ctx, gc, (GraphEntity *)e);
}

static void _ResultSet_CompactReplyWithSIArray
(
	RedisModuleCtx *ctx,
	GraphContext *gc,
	SIValue array
) {

	/*  Compact array reply format:
	 *  [
	 *      [type, value] // every member is returned at its compact representation
	 *      [type, value]
	 *      .
	 *      .
	 *      .
	 *      [type, value]
	 *  ]
	 */
	uint arrayLen = SIArray_Length(array);
	RedisModule_ReplyWithArray(ctx, arrayLen);
	for(uint i = 0; i < arrayLen; i++) {
		RedisModule_ReplyWithArray(ctx, 2); // Reply with array with space for type and value
		_ResultSet_CompactReplyWithSIValue(ctx, gc, SIArray_Get(array, i));
	}
}

static void _ResultSet_CompactReplyWithVector32F
(
	RedisModuleCtx *ctx,
	SIValue vec
) {
	/*  Compact vector reply format:
	 *  [
	 *      value
	 *      .
	 *      .
	 *      .
	 *      value
	 *  ]
	 */

	ASSERT(SI_TYPE(vec) == T_VECTOR_F32);

	// construct arrry of vector elements
	uint32_t dim = SIVector_Dim(vec);
	RedisModule_ReplyWithArray(ctx, dim);

	// get vector elements
	void *elements = SIVector_Elements(vec);

	// reply with vector elements
	float *values = (float*)elements;
	for(uint i = 0; i < dim; i++) {
		RedisModule_ReplyWithDouble(ctx, (double)values[i]);
	}
}

static void _ResultSet_CompactReplyWithPath
(
	RedisModuleCtx *ctx,
	GraphContext *gc,
	SIValue path
) {
	/* Path will return as an array of two SIArrays, the first is path nodes and the second is edges,
	* see array compact format.
	* Compact path reply:
	* [
	*      type : array,
	*      [
	*          [Node compact reply format],
	*          .
	*          .
	*          .
	*          [Node compact reply format]
	*      ],
	*      type: array,
	*      [
	*          [Edge compact reply format],
	*          .
	*          .
	*          .
	*          [Edge compact reply format]
	*      ]
	* ]
	*/

	// Response consists of two arrays.
	RedisModule_ReplyWithArray(ctx, 2);
	// First array type and value.
	RedisModule_ReplyWithArray(ctx, 2);
	SIValue nodes = SIPath_Nodes(path);
	_ResultSet_CompactReplyWithSIValue(ctx, gc, nodes);
	SIValue_Free(nodes);
	// Second array type and value.
	RedisModule_ReplyWithArray(ctx, 2);
	SIValue relationships = SIPath_Relationships(path);
	_ResultSet_CompactReplyWithSIValue(ctx, gc, relationships);
	SIValue_Free(relationships);
}

static void _ResultSet_CompactReplyWithMap
(
	RedisModuleCtx *ctx,
	GraphContext *gc,
	SIValue v
) {
	// map will be returned as an array of key/value pairs
	// consider the map object: {a:1, b:'str', c: {x:1, y:2}}
	//
	// the reply will be structured:
	// [
	//     string(a), int(1),
	//     string(b), string(str),
	//     string(c), [
	//                    string(x), int(1),
	//                    string(y), int(2)
	//                 ]
	// ]

	uint key_count = Map_KeyCount(v);
	Map m = v.map;

	// response consists of N pairs array:
	// (string, value type, value)
	RedisModule_ReplyWithArray(ctx, key_count * 2);
	for(int i = 0; i < key_count; i++) {
		Pair     p     =  m[i];
		SIValue  val   =  p.val;
		char     *key  =  p.key.stringval;

		// emit key
		RedisModule_ReplyWithCString(ctx, key);

		// emit value
		RedisModule_ReplyWithArray(ctx, 2);
		_ResultSet_CompactReplyWithSIValue(ctx, gc, val);
	}
}

static void _ResultSet_CompactReplyWithPoint
(
	RedisModuleCtx *ctx,
	GraphContext *gc,
	SIValue v
) {
	ASSERT(SI_TYPE(v) == T_POINT);
	RedisModule_ReplyWithArray(ctx, 2);

	_ResultSet_ReplyWithRoundedDouble(ctx, Point_lat(v));
	_ResultSet_ReplyWithRoundedDouble(ctx, Point_lon(v));
}

void ResultSet_EmitCompactRow
(
	ResultSet *set,
	SIValue **row
) {
	RedisModuleCtx *ctx = set->ctx;
	// Prepare return array sized to the number of RETURN entities
	RedisModule_ReplyWithArray(ctx, set->column_count);

	for(uint i = 0; i < set->column_count; i++) {
		SIValue cell = *row[i];
		RedisModule_ReplyWithArray(ctx, 2); // Reply with array with space for type and value
		_ResultSet_CompactReplyWithSIValue(ctx, set->gc, cell);
	}
}

// For every column in the header, emit a 2-array containing the ColumnType enum
// followed by the column alias.
void ResultSet_ReplyWithCompactHeader
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
		RedisModule_ReplyWithArray(ctx, 2);
		// because the types found in the first Record do not necessarily inform the types
		// in subsequent records, we will always set the column type as scalar
		ColumnType t = COLUMN_SCALAR;
		RedisModule_ReplyWithLongLong(ctx, t);

		// Second, emit the identifier string associated with the column
		RedisModule_ReplyWithStringBuffer(ctx, set->columns[i], strlen(set->columns[i]));
	}
}

void ResultSet_EmitCompactStats
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

	RedisModule_ReplyWithArray(set->ctx, resultset_size);

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

