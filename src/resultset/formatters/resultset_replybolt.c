/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "resultset_formatters.h"
#include "../../util/arr.h"
#include "../../datatypes/datatypes.h"
#include "../../bolt/bolt.h"
#include "../../bolt/socket.h"
#include "../../commands/cmd_context.h"
#include "../../globals.h"

void _ResultSet_BoltReplyWithSIValue(bolt_client_t *client, GraphContext *gc, SIValue v) {
	switch(SI_TYPE(v)) {
	case T_STRING:
		bolt_reply_string(client, v.stringval);
		break;
	case T_INT64:
		if(v.longval < UINT8_MAX) {
			bolt_reply_int8(client, v.longval);
		}
		else if(v.longval < UINT16_MAX) {
			bolt_reply_int16(client, v.longval);
		}
		else if(v.longval < UINT32_MAX) {
			bolt_reply_int32(client, v.longval);
		} else {
			bolt_reply_int64(client, v.longval);
		}
		break;
	case T_DOUBLE:
		bolt_reply_float(client, v.doubleval);
		break;
	case T_BOOL:
		bolt_reply_bool(client, v.longval);
		break;
	case T_NULL:
		bolt_reply_null(client);
		break;
	case T_NODE: {
		Node *n = v.ptrval;
		bolt_reply_structure(client, BST_NODE, 4);
		bolt_reply_int64(client, n->id);
		uint lbls_count;
		NODE_GET_LABELS(gc->g, n, lbls_count);
		bolt_reply_list(client, lbls_count);
		for(int i = 0; i < lbls_count; i++) {
			Schema *s = GraphContext_GetSchemaByID(gc, labels[i], SCHEMA_NODE);
			const char *lbl_name = Schema_GetName(s);
			bolt_reply_string(client, lbl_name);
		}
		const AttributeSet set = GraphEntity_GetAttributes((GraphEntity *)n);
		int prop_count = ATTRIBUTE_SET_COUNT(set);
		bolt_reply_map(client, prop_count);
		// Iterate over all properties stored on entity
		for(int i = 0; i < prop_count; i ++) {
			Attribute_ID attr_id;
			SIValue value = AttributeSet_GetIdx(set, i, &attr_id);
			// Emit the actual string
			const char *prop_str = GraphContext_GetAttributeString(gc, attr_id);
			bolt_reply_string(client, prop_str);
			// Emit the value
			_ResultSet_BoltReplyWithSIValue(client, gc, value);
		}
		bolt_reply_string(client, "node_0");
		break;
	}
	case T_EDGE: {
		Edge *e = v.ptrval;
		bolt_reply_structure(client, BST_RELATIONSHIP, 8);
		bolt_reply_int64(client, e->id);
		bolt_reply_int64(client, e->src_id);
		bolt_reply_int64(client, e->dest_id);
		Schema *s = GraphContext_GetSchemaByID(gc, Edge_GetRelationID(e), SCHEMA_EDGE);
		const char *reltype = Schema_GetName(s);
		bolt_reply_string(client, reltype);
		const AttributeSet set = GraphEntity_GetAttributes((GraphEntity *)e);
		int prop_count = ATTRIBUTE_SET_COUNT(set);
		bolt_reply_map(client, prop_count);
		// Iterate over all properties stored on entity
		for(int i = 0; i < prop_count; i ++) {
			Attribute_ID attr_id;
			SIValue value = AttributeSet_GetIdx(set, i, &attr_id);
			// Emit the actual string
			const char *prop_str = GraphContext_GetAttributeString(gc, attr_id);
			bolt_reply_string(client, prop_str);
			// Emit the value
			_ResultSet_BoltReplyWithSIValue(client, gc, value);
		}
		bolt_reply_string(client, "relationship_0");
		bolt_reply_string(client, "node_0");
		bolt_reply_string(client, "node_0");
		break;
	}
	case T_ARRAY:
		bolt_reply_list(client, SIArray_Length(v));
		for(int i = 0; i < SIArray_Length(v); i++) {
			_ResultSet_BoltReplyWithSIValue(client, gc, SIArray_Get(v, i));
		}
		break;
	case T_PATH:
		bolt_reply_null(client);
		break;
	case T_MAP:
		bolt_reply_map(client, Map_KeyCount(v));
		for(uint i = 0; i < Map_KeyCount(v); i ++) {
			Pair p = v.map[i];
			_ResultSet_BoltReplyWithSIValue(client, gc, p.key);
			_ResultSet_BoltReplyWithSIValue(client, gc, p.val);
		}
		break;
	case T_POINT:
		bolt_reply_null(client);
		break;
	default:
		RedisModule_Assert("Unhandled value type" && false);
	}
}

void ResultSet_EmitBoltRow
(
	RedisModuleCtx *ctx,
	bolt_client_t *bolt_client,
	GraphContext *gc,
	SIValue **row,
	uint numcols
) {
	bolt_reply_structure(bolt_client, BST_RECORD, 1);
	bolt_reply_list(bolt_client, numcols);
	for(int i = 0; i < numcols; i++) {
		_ResultSet_BoltReplyWithSIValue(bolt_client, gc, *row[i]);
	}
	bolt_client_send(bolt_client);
}

// Emit the alias or descriptor for each column in the header.
void ResultSet_ReplyWithBoltHeader
(
	RedisModuleCtx *ctx,
	bolt_client_t *bolt_client,
	const char **columns,
	uint *col_rec_map
) {
	bolt_reply_structure(bolt_client, BST_SUCCESS, 1);
	bolt_reply_map(bolt_client, 3);
	bolt_reply_string(bolt_client, "t_first");
	bolt_reply_int8(bolt_client, 2);
	bolt_reply_string(bolt_client, "fields");
	bolt_reply_list(bolt_client, array_len(columns));
	for(int i = 0; i < array_len(columns); i++) {
		bolt_reply_string(bolt_client, columns[i]);
	}
	bolt_reply_string(bolt_client, "qid");
	bolt_reply_int8(bolt_client, 0);
	bolt_client_send(bolt_client);
}
