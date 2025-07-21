/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "resultset_formatters.h"
#include "RG.h"
#include "../../util/arr.h"
#include "../../datatypes/datatypes.h"

// Forward declarations.
static void _ResultSet_CompactReplyWithNode(RedisModuleCtx *ctx, GraphContext *gc, Node *n);
static void _ResultSet_CompactReplyWithEdge(RedisModuleCtx *ctx, GraphContext *gc, Edge *e);
static void _ResultSet_CompactReplyWithSIArray(RedisModuleCtx *ctx, GraphContext *gc, SIValue array);
static void _ResultSet_CompactReplyWithPath(RedisModuleCtx *ctx, GraphContext *gc, SIValue path);
static void _ResultSet_CompactReplyWithMap(RedisModuleCtx *ctx, GraphContext *gc, SIValue v);
static void _ResultSet_CompactReplyWithPoint(RedisModuleCtx *ctx, GraphContext *gc, SIValue v);

static inline ValueType _mapValueType(const SIValue v) {
	switch(SI_TYPE(v)) {
	case T_NULL:
		return VALUE_NULL;
	case T_STRING:
		return VALUE_STRING;
	case T_INT64:
		return VALUE_INTEGER;
	case T_BOOL:
		return VALUE_BOOLEAN;
	case T_DOUBLE:
		return VALUE_DOUBLE;
	case T_ARRAY:
		return VALUE_ARRAY;
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
	default:
		return VALUE_UNKNOWN;
	}
}

static inline void _ResultSet_ReplyWithValueType(RedisModuleCtx *ctx, const SIValue v) {
	RedisModule_ReplyWithLongLong(ctx, _mapValueType(v));
}

static void _ResultSet_CompactReplyWithSIValue(RedisModuleCtx *ctx, GraphContext *gc,
											   const SIValue v) {
	// Emit the value type, then the actual value (to facilitate client-side parsing)
	_ResultSet_ReplyWithValueType(ctx, v);

	switch(SI_TYPE(v)) {
	case T_STRING:
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
	case T_ARRAY:
		_ResultSet_CompactReplyWithSIArray(ctx, gc, v);
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

static void _ResultSet_CompactReplyWithProperties(RedisModuleCtx *ctx, GraphContext *gc,
												  const GraphEntity *e) {
	const AttributeSet set = GraphEntity_GetAttributes(e);
	int prop_count = AttributeSet_Count(set);
	RedisModule_ReplyWithArray(ctx, prop_count);
	// Iterate over all properties stored on entity
	for(int i = 0; i < prop_count; i ++) {
		// Compact replies include the value's type; verbose replies do not
		RedisModule_ReplyWithArray(ctx, 3);
		Attribute_ID attr_id;
		SIValue value = AttributeSet_GetIdx(set, i, &attr_id);
		// Emit the string index
		RedisModule_ReplyWithLongLong(ctx, attr_id);
		// Emit the value
		_ResultSet_CompactReplyWithSIValue(ctx, gc, value);
	}
}

static void _ResultSet_CompactReplyWithNode(RedisModuleCtx *ctx, GraphContext *gc, Node *n) {
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

static void _ResultSet_CompactReplyWithEdge(RedisModuleCtx *ctx, GraphContext *gc, Edge *e) {
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

static void _ResultSet_CompactReplyWithSIArray(RedisModuleCtx *ctx, GraphContext *gc,
											   SIValue array) {

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

static void _ResultSet_CompactReplyWithPath(RedisModuleCtx *ctx, GraphContext *gc, SIValue path) {
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

static void _ResultSet_CompactReplyWithMap(RedisModuleCtx *ctx, GraphContext *gc, SIValue v) {
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

static void _ResultSet_CompactReplyWithPoint(RedisModuleCtx *ctx, GraphContext *gc, SIValue v) {
	ASSERT(SI_TYPE(v) == T_POINT);
	RedisModule_ReplyWithArray(ctx, 2);

	_ResultSet_ReplyWithRoundedDouble(ctx, Point_lat(v));
	_ResultSet_ReplyWithRoundedDouble(ctx, Point_lon(v));
}

void ResultSet_EmitCompactRow(RedisModuleCtx *ctx, GraphContext *gc,
							  SIValue **row, uint numcols) {
	// Prepare return array sized to the number of RETURN entities
	RedisModule_ReplyWithArray(ctx, numcols);

	for(uint i = 0; i < numcols; i++) {
		SIValue cell = *row[i];
		RedisModule_ReplyWithArray(ctx, 2); // Reply with array with space for type and value
		_ResultSet_CompactReplyWithSIValue(ctx, gc, cell);
	}
}

// For every column in the header, emit a 2-array containing the ColumnType enum
// followed by the column alias.
void ResultSet_ReplyWithCompactHeader(RedisModuleCtx *ctx, const char **columns,
									  uint *col_rec_map) {
	uint columns_len = array_len(columns);
	RedisModule_ReplyWithArray(ctx, columns_len);
	for(uint i = 0; i < columns_len; i++) {
		RedisModule_ReplyWithArray(ctx, 2);
		/* Because the types found in the first Record do not necessarily inform the types
		 * in subsequent records, we will always set the column type as scalar. */
		ColumnType t = COLUMN_SCALAR;
		RedisModule_ReplyWithLongLong(ctx, t);

		// Second, emit the identifier string associated with the column
		RedisModule_ReplyWithStringBuffer(ctx, columns[i], strlen(columns[i]));
	}
}
