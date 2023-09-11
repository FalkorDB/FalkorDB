/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "resultset_formatters.h"
#include "../../datatypes/datatypes.h"

static void _ResultSet_BoltReplyWithNode
(
	bolt_client_t *client,
	GraphContext *gc,
	Node *n
);

static void _ResultSet_BoltReplyWithEdge
(
	bolt_client_t *client,
	GraphContext *gc,
	Edge *e,
	bool is_bounded
);

static void _ResultSet_BoltReplyWithPath
(
	bolt_client_t *client,
	GraphContext *gc,
	SIValue path
);

void _ResultSet_BoltReplyWithSIValue
(
	bolt_client_t *client,
	GraphContext *gc,
	SIValue v
) {
	switch(SI_TYPE(v)) {
	case T_STRING:
		bolt_reply_string(client, v.stringval, strlen(v.stringval));
		break;
	case T_INT64:
		bolt_reply_int(client, v.longval);
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
	case T_NODE:
		_ResultSet_BoltReplyWithNode(client, gc, v.ptrval);
		break;
	case T_EDGE:
		_ResultSet_BoltReplyWithEdge(client, gc, v.ptrval, true);
		break;
	case T_ARRAY:
		bolt_reply_list(client, SIArray_Length(v));
		for(int i = 0; i < SIArray_Length(v); i++) {
			_ResultSet_BoltReplyWithSIValue(client, gc, SIArray_Get(v, i));
		}
		break;
	case T_PATH:
		_ResultSet_BoltReplyWithPath(client, gc, v);
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
		// Point2D::Structure(
		//     srid::Integer,
		//     x::Float,
		//     y::Float,
		// )
		bolt_reply_structure(client, BST_POINT2D, 3);
		bolt_reply_int64(client, 4326);
		bolt_reply_float(client, v.point.longitude);
		bolt_reply_float(client, v.point.latitude);
		bolt_reply_null(client);
		break;
	case T_VECTOR32F:
		uint32_t dim = SIVector_Dim(v);
		bolt_reply_list(client, dim);

		// get vector elements
		void *elements = SIVector_Elements(v);

		// reply with vector elements
		float *values = (float*)elements;
		for(uint i = 0; i < dim; i++) {
			bolt_reply_float(client, (double)values[i]);
		}
		break;
	default:
		RedisModule_Assert("Unhandled value type" && false);
	}
}

static void _ResultSet_BoltReplyWithElementID
(
	bolt_client_t *client,
	uint64_t id,
	const char *prefix
) {
	int ndigits = id == 0 ? 1 : floor(log10(id)) + 1;
	char element_id[ndigits + strlen(prefix) + 2];
	sprintf(element_id, "%s_%llu", prefix, id);
	bolt_reply_string(client, element_id, strlen(element_id));
}

static void _ResultSet_BoltReplyWithNode
(
	bolt_client_t *client,
	GraphContext *gc,
	Node *n
) {
	// Node::Structure(
	//     id::Integer,
	//     labels::List<String>,
	//     properties::Dictionary,
	//     element_id::String,
	// )

	bolt_reply_structure(client, BST_NODE, 4);
	bolt_reply_int64(client, n->id);
	uint lbls_count;
	NODE_GET_LABELS(gc->g, n, lbls_count);
	bolt_reply_list(client, lbls_count);
	for(int i = 0; i < lbls_count; i++) {
		Schema *s = GraphContext_GetSchemaByID(gc, labels[i], SCHEMA_NODE);
		const char *lbl_name = Schema_GetName(s);
		bolt_reply_string(client, lbl_name, strlen(lbl_name));
	}
	const AttributeSet set = GraphEntity_GetAttributes((GraphEntity *)n);
	int prop_count = AttributeSet_Count(set);
	bolt_reply_map(client, prop_count);
	// Iterate over all properties stored on entity
	for(int i = 0; i < prop_count; i ++) {
		Attribute_ID attr_id;
		SIValue value = AttributeSet_GetIdx(set, i, &attr_id);
		// Emit the actual string
		const char *prop_str = GraphContext_GetAttributeString(gc, attr_id);
		bolt_reply_string(client, prop_str, strlen(prop_str));
		// Emit the value
		_ResultSet_BoltReplyWithSIValue(client, gc, value);
	}
	_ResultSet_BoltReplyWithElementID(client, n->id, "node");
}

static void _ResultSet_BoltReplyWithEdge
(
	bolt_client_t *client,
	GraphContext *gc,
	Edge *e,
	bool is_bounded
) {
	// Relationship::Structure(
	//     id::Integer,
	//     startNodeId::Integer,
	//     endNodeId::Integer,
	//     type::String,
	//     properties::Dictionary,
	//     element_id::String,
	//     start_node_element_id::String,
	//     end_node_element_id::String,
	// )

	// UnboundRelationship::Structure(
	//     id::Integer,
	//     type::String,
	//     properties::Dictionary,
	//     element_id::String,
	// )

	if(is_bounded) {
		bolt_reply_structure(client, BST_RELATIONSHIP, 8);
	} else {
		bolt_reply_structure(client, BST_UNBOUND_RELATIONSHIP, 4);
	}
	
	bolt_reply_int64(client, e->id);
	if(is_bounded) {
		bolt_reply_int64(client, e->src_id);
		bolt_reply_int64(client, e->dest_id);
	}
	Schema *s = GraphContext_GetSchemaByID(gc, Edge_GetRelationID(e), SCHEMA_EDGE);
	const char *reltype = Schema_GetName(s);
	bolt_reply_string(client, reltype, strlen(reltype));
	const AttributeSet set = GraphEntity_GetAttributes((GraphEntity *)e);
	int prop_count = AttributeSet_Count(set);
	bolt_reply_map(client, prop_count);
	// Iterate over all properties stored on entity
	for(int i = 0; i < prop_count; i ++) {
		Attribute_ID attr_id;
		SIValue value = AttributeSet_GetIdx(set, i, &attr_id);
		// Emit the actual string
		const char *prop_str = GraphContext_GetAttributeString(gc, attr_id);
		bolt_reply_string(client, prop_str, strlen(prop_str));
		// Emit the value
		_ResultSet_BoltReplyWithSIValue(client, gc, value);
	}
	_ResultSet_BoltReplyWithElementID(client, e->id, "relationship");
	if(is_bounded) {
		_ResultSet_BoltReplyWithElementID(client, e->src_id, "node");
		_ResultSet_BoltReplyWithElementID(client, e->dest_id, "node");
	}
}

static void _ResultSet_BoltReplyWithPath
(
	bolt_client_t *client,
	GraphContext *gc,
	SIValue path
) {
	// Path::Structure(
	//     nodes::List<Node>,
	//     rels::List<UnboundRelationship>,
	//     indices::List<Integer>,
	// )

	bolt_reply_structure(client, BST_PATH, 3);
	size_t node_count = SIPath_NodeCount(path);
	bolt_reply_list(client, node_count);
	for(int i = 0; i < node_count; i++) {
		_ResultSet_BoltReplyWithNode(client, gc, SIPath_GetNode(path, i).ptrval);
	}
	size_t edge_count = SIPath_EdgeCount(path);
	bolt_reply_list(client, edge_count);
	for(int i = 0; i < edge_count; i++) {
		_ResultSet_BoltReplyWithEdge(client, gc, SIPath_GetRelationship(path, i).ptrval, false);
	}

	size_t indices = node_count + edge_count - 1;
	bolt_reply_list(client, indices);
	for(uint8_t i = 0; i < edge_count; i++) {
		Edge *e = (Edge *)SIPath_GetRelationship(path, i).ptrval;
		Node *n = (Node *)SIPath_GetNode(path, i).ptrval;
		if(e->src_id == n->id) {
			bolt_reply_int8(client, i + 1);
		} else {
			bolt_reply_int8(client, -(i + 1));
		}
		bolt_reply_int8(client, i + 1);
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
	bolt_reply_string(bolt_client, "t_first", 7);
	bolt_reply_int8(bolt_client, 2);
	bolt_reply_string(bolt_client, "fields", 6);
	bolt_reply_list(bolt_client, array_len(columns));
	for(int i = 0; i < array_len(columns); i++) {
		bolt_reply_string(bolt_client, columns[i], strlen(columns[i]));
	}
	bolt_reply_string(bolt_client, "qid", 3);
	bolt_reply_int8(bolt_client, 0);
	bolt_client_finish_write(bolt_client);
}
