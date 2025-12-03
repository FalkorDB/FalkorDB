/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "../resultset.h"
#include "../../bolt/bolt.h"
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
	case T_INTERN_STRING:
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
	case T_VECTOR_F32: {
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
	}
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
	sprintf(element_id, "%s_%" PRIu64, prefix, id);
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
		SIValue value ;
		AttributeID attr_id ;
		AttributeSet_GetIdx (set, i, &attr_id, &value) ;

		// emit the actual string
		const char *prop_str = GraphContext_GetAttributeString(gc, attr_id);
		bolt_reply_string(client, prop_str, strlen(prop_str));

		// emit the value
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
		SIValue value ;
		AttributeID attr_id ;
		AttributeSet_GetIdx (set, i, &attr_id, &value) ;

		// emit the actual string
		const char *prop_str = GraphContext_GetAttributeString(gc, attr_id);
		bolt_reply_string(client, prop_str, strlen(prop_str));

		// emit the value
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
	for(size_t i = 0; i < edge_count; i++) {
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
	ResultSet *set,
	SIValue **row
) {
	bolt_client_t *bolt_client = set->bolt_client;
	bolt_client_reply_for(set->bolt_client, BST_PULL, BST_RECORD, 1);
	bolt_reply_list(set->bolt_client, set->column_count);
	for(int i = 0; i < set->column_count; i++) {
		_ResultSet_BoltReplyWithSIValue(bolt_client, set->gc, *row[i]);
	}
	bolt_client_end_message(bolt_client);
}

// Emit the alias or descriptor for each column in the header.
void ResultSet_ReplyWithBoltHeader
(
	ResultSet *set
) {
	bolt_client_t *bolt_client = set->bolt_client;
	bolt_client_reply_for(bolt_client, BST_RUN, BST_SUCCESS, 1);
	bolt_reply_map(bolt_client, 3);
	bolt_reply_string(bolt_client, "t_first", 7);
	bolt_reply_int8(bolt_client, 2);
	bolt_reply_string(bolt_client, "fields", 6);
	bolt_reply_list(bolt_client, set->column_count);
	for(int i = 0; i < set->column_count; i++) {
		bolt_reply_string(bolt_client, set->columns[i], strlen(set->columns[i]));
	}
	bolt_reply_string(bolt_client, "qid", 3);
	bolt_reply_int8(bolt_client, 0);
	bolt_client_end_message(bolt_client);
}

void ResultSet_EmitBoltStats
(
	ResultSet *set
) {
	bolt_client_reply_for(set->bolt_client, BST_PULL, BST_SUCCESS, 1);
	int stats = 0;
	if(set->stats.index_creation)            stats++;
	if(set->stats.index_deletion)            stats++;
	if(set->stats.constraint_creation)       stats++;
	if(set->stats.constraint_deletion)       stats++;
	if(set->stats.labels_added          > 0) stats++;
	if(set->stats.nodes_created         > 0) stats++;
	if(set->stats.nodes_deleted         > 0) stats++;
	if(set->stats.labels_removed        > 0) stats++;
	if(set->stats.properties_set        > 0) stats++;
	if(set->stats.properties_removed    > 0) stats++;
	if(set->stats.relationships_deleted > 0) stats++;
	if(set->stats.relationships_created > 0) stats++;
	if(stats > 0) {
		bolt_reply_map(set->bolt_client, 2);
		bolt_reply_string(set->bolt_client, "stats", 5);
		bolt_reply_map(set->bolt_client, stats);
		if(set->stats.index_creation) {
			bolt_reply_string(set->bolt_client, "indexes-added", 13);
			bolt_reply_int(set->bolt_client, set->stats.indices_created);
		}
		if(set->stats.index_deletion) {
			bolt_reply_string(set->bolt_client, "indexes-removed", 15);
			bolt_reply_int(set->bolt_client, set->stats.indices_deleted);
		}
		if(set->stats.constraint_creation) {
			bolt_reply_string(set->bolt_client, "constraints-added", 17);
			bolt_reply_int(set->bolt_client, set->stats.constraints_created);
		}
		if(set->stats.constraint_deletion) {
			bolt_reply_string(set->bolt_client, "constraints-removed", 19);
			bolt_reply_int(set->bolt_client, set->stats.constraints_deleted);
		}
		if(set->stats.labels_added          > 0) {
			bolt_reply_string(set->bolt_client, "labels-added", 12);
			bolt_reply_int(set->bolt_client, set->stats.labels_added);
		}
		if(set->stats.nodes_created         > 0) {
			bolt_reply_string(set->bolt_client, "nodes-created", 13);
			bolt_reply_int(set->bolt_client, set->stats.nodes_created);
		}
		if(set->stats.nodes_deleted         > 0) {
			bolt_reply_string(set->bolt_client, "nodes-deleted", 13);
			bolt_reply_int(set->bolt_client, set->stats.nodes_deleted);
		}
		if(set->stats.labels_removed        > 0) {
			bolt_reply_string(set->bolt_client, "labels-removed", 14);
			bolt_reply_int(set->bolt_client, set->stats.labels_removed);
		}
		if(set->stats.properties_set        > 0) {
			bolt_reply_string(set->bolt_client, "properties-set", 14);
			bolt_reply_int(set->bolt_client, set->stats.properties_set);
		}
		if(set->stats.properties_removed    > 0) {
			bolt_reply_string(set->bolt_client, "properties-removed", 18);
			bolt_reply_int(set->bolt_client, set->stats.properties_removed);
		}
		if(set->stats.relationships_deleted > 0) {
			bolt_reply_string(set->bolt_client, "relationships-deleted", 21);
			bolt_reply_int(set->bolt_client, set->stats.relationships_deleted);
		}
		if(set->stats.relationships_created > 0) {
			bolt_reply_string(set->bolt_client, "relationships-created", 21);
			bolt_reply_int(set->bolt_client, set->stats.relationships_created);
		}
	} else {
		bolt_reply_map(set->bolt_client, 1);
	}
	bolt_reply_string(set->bolt_client, "t_last", 6);
	bolt_reply_int8(set->bolt_client, 1);

	bolt_client_end_message(set->bolt_client);
	bolt_client_finish_write(set->bolt_client);
}
