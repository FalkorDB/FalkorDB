/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "json_encoder.h"
#include "sds/sds.h"
#include "rmalloc.h"
#include "strutil.h"
#include "../value.h"
#include "../query_ctx.h"
#include "../errors/errors.h"
#include "../graph/graphcontext.h"
#include "../graph/entities/node.h"
#include "../graph/entities/edge.h"
#include "../datatypes/datatypes.h"

// Forward declaration
sds _JsonEncoder_SIValue(SIValue v, sds s);

static inline sds _JsonEncoder_String(SIValue v, sds s) {
	return sdscatfmt(s, "\"%s\"", v.stringval);
}

static sds _JsonEncoder_Properties
(
	const GraphEntity *ge,
	sds s
) {
	s = sdscat (s, "\"properties\": {") ;
	const AttributeSet set = GraphEntity_GetAttributes (ge) ;
	uint prop_count = AttributeSet_Count (set) ;
	GraphContext *gc = QueryCtx_GetGraphCtx () ;

	for (uint i = 0; i < prop_count; i ++) {
		SIValue value ;
		AttributeID attr_id ;
		AttributeSet_GetIdx (set, i, &attr_id, &value) ;

		const char *key = GraphContext_GetAttributeString (gc, attr_id) ;
		s = sdscatfmt (s, "\"%s\": ", key) ;
		s = _JsonEncoder_SIValue (value, s) ;
		if (i < prop_count - 1) {
			s = sdscat (s, ", ") ;
		}
	}

	s = sdscat (s, "}") ;
	return s ;
}

static sds _JsonEncoder_Node(const Node *n, sds s) {
	s = sdscatfmt(s, "\"id\": %U", ENTITY_GET_ID(n));
	s = sdscat(s, ", \"labels\": [");
	// Retrieve node labels
	uint label_count;
	GraphContext *gc = QueryCtx_GetGraphCtx();
	NODE_GET_LABELS(gc->g, n, label_count);
	for(uint i = 0; i < label_count; i ++) {
		Schema *schema = GraphContext_GetSchemaByID(gc, labels[i], SCHEMA_NODE);
		ASSERT(schema);
		const char *label = Schema_GetName(schema);
		ASSERT(label);
		s = sdscatfmt(s, "\"%s\"", label);
		if(i != label_count - 1) s = sdscat(s, ", ");
	}
	s = sdscat(s, "], ");
	s = _JsonEncoder_Properties((const GraphEntity *)n, s);
	return s;
}

static sds _JsonEncoder_Edge(Edge *e, sds s) {
	s = sdscatfmt(s, "\"id\": %U", ENTITY_GET_ID(e));
	GraphContext *gc = QueryCtx_GetGraphCtx();
	// Retrieve reltype data.
	int id = Edge_GetRelationID(e);
	Schema *schema = GraphContext_GetSchemaByID(gc, id, SCHEMA_EDGE);
	ASSERT(schema);
	const char *relationship = Schema_GetName(schema);
	ASSERT(relationship);
	s = sdscatfmt(s, ", \"relationship\": \"%s\", ", relationship);

	s = _JsonEncoder_Properties((const GraphEntity *)e, s);

	s = sdscat(s, ", \"start\": {");
	// Retrieve source node data.
	Node src;
	Graph_GetNode(gc->g, e->src_id, &src);
	s = _JsonEncoder_Node(&src, s);

	s = sdscat(s, "}, \"end\": {");
	// Retrieve dest node data.
	Node dest;
	Graph_GetNode(gc->g, e->dest_id, &dest);
	s = _JsonEncoder_Node(&dest, s);

	s = sdscat(s, "}");
	return s;
}

static sds _JsonEncoder_GraphEntity(GraphEntity *ge, sds s, GraphEntityType type) {
	switch(type) {
	case GETYPE_NODE:
		s = sdscat(s, "{\"type\": \"node\", ");
		s = _JsonEncoder_Node((const Node *)ge, s);
		break;
	case GETYPE_EDGE:
		s = sdscat(s, "{\"type\": \"relationship\", ");
		s = _JsonEncoder_Edge((Edge *)ge, s);
		break;
	default:
		ASSERT(false);
	}
	s = sdscat(s, "}");
	return s;
}

static sds _JsonEncoder_Path(SIValue p, sds s) {
	// open path with "["
	s = sdscat(s, "[");

	size_t nodeCount = SIPath_NodeCount(p);
	for(size_t i = 0; i < nodeCount - 1; i ++) {
		// write the next value
		SIValue node = SIPath_GetNode(p, i);
		s = _JsonEncoder_GraphEntity((GraphEntity *)node.ptrval, s, GETYPE_NODE);
		s = sdscat(s, ", ");
		SIValue edge = SIPath_GetRelationship(p, i);
		s = _JsonEncoder_GraphEntity((GraphEntity *)edge.ptrval, s, GETYPE_EDGE);
		s = sdscat(s, ", ");
	}
	// Handle last node.
	if(nodeCount > 0) {
		SIValue node = SIPath_GetNode(p, nodeCount - 1);
		s = _JsonEncoder_GraphEntity((GraphEntity *)node.ptrval, s, GETYPE_NODE);
	}

	// close array with "]"
	s = sdscat(s, "]");
	return s;
}

static sds _JsonEncoder_Point(SIValue point, sds s) {
	ASSERT(SI_TYPE(point) & T_POINT);

	// default crs == wgs-84 till we support other CRS formats 
	s = sdscat(s, "{\"crs\":\"wgs-84\",\"latitude\":");

	s = sdscatprintf(s, "%f", Point_lat(point));
	s = sdscat(s, ",\"longitude\":");
	s = sdscatprintf(s, "%f", Point_lon(point));

	// height is not supported yet
	s = sdscat(s, ",\"height\":null");
	s = sdscat(s, "}");
	return s;
}

static sds _JsonEncoder_Array(SIValue list, sds s) {
	// open array with "["
	s = sdscat(s, "[");
	uint arrayLen = SIArray_Length(list);
	for(uint i = 0; i < arrayLen; i ++) {
		// write the next value
		s = _JsonEncoder_SIValue(SIArray_Get(list, i), s);
		// if it is not the last element, add ", "
		if(i != arrayLen - 1) s = sdscat(s, ", ");
	}

	// close array with "]"
	s = sdscat(s, "]");
	return s;
}

static sds _JsonEncoder_Map(SIValue map, sds s) {
	ASSERT(SI_TYPE(map) & T_MAP);

	// "{" marks the beginning of a map
	s = sdscat(s, "{");

	uint key_count = Map_KeyCount(map);
	for(uint i = 0; i < key_count; i ++) {
		Pair p = map.map[i];
		// write the next key/value pair
		s = _JsonEncoder_String(p.key, s);
		s = sdscat(s, ": ");
		s = _JsonEncoder_SIValue(p.val, s);
		// if this is not the last element, add ", "
		if(i != key_count - 1) s = sdscat(s, ", ");
	}

	// "}" marks the end of a map
	s = sdscat(s, "}");
	return s;
}

sds _JsonEncoder_SIValue(SIValue v, sds s) {
	switch(v.type) {
	case T_STRING:
	case T_INTERN_STRING:
		s = _JsonEncoder_String(v, s);
		break;
	case T_INT64:
		s = sdscatfmt(s, "%I", v.longval);
		break;
	case T_BOOL:
		if(v.longval) s = sdscat(s, "true");
		else s = sdscat(s, "false");
		break;
	case T_DOUBLE:
		s = sdscatprintf(s, "%.15g", v.doubleval);
		break;
	case T_NODE:
		s = _JsonEncoder_GraphEntity(v.ptrval, s, GETYPE_NODE);
		break;
	case T_EDGE:
		s = _JsonEncoder_GraphEntity(v.ptrval, s, GETYPE_EDGE);
		break;
	case T_ARRAY:
		s = _JsonEncoder_Array(v, s);
		break;
	case T_MAP:
		s = _JsonEncoder_Map(v, s);
		break;
	case T_PATH:
		s = _JsonEncoder_Path(v, s);
		break;
	case T_NULL:
		s = sdscat(s, "null");
		break;
	case T_POINT:
		s = _JsonEncoder_Point(v, s);
		break;		
	default:
		// unrecognized type
		ErrorCtx_RaiseRuntimeException("JSON encoder encountered unrecognized type: %d\n", v.type);
		ASSERT(false);
		break;
	}
	return s;
}

char *JsonEncoder_SIValue(SIValue v) {
	// Create an empty sds string.
	sds s = sdsempty();
	// Populate the sds string with encoded data.
	s = _JsonEncoder_SIValue(v, s);
	// Duplicate the sds string into a standard C string.
	char *retval = rm_strdup(s);
	// Free the sds string.
	sdsfree(s);
	return retval;
}

