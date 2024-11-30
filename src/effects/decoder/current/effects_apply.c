/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../effects.h"
#include "../../../graph/graph_hub.h"

#include <stdio.h>

// read effect type from stream
static inline EffectType ReadEffectType
(
	SerializerIO stream  // effects stream
) {
	// read EffectType off of stream
	EffectType t = SerializerIO_ReadUnsigned(stream);

	return t;
}

static AttributeSet ReadAttributeSet
(
	SerializerIO stream  // effects stream
) {
	//--------------------------------------------------------------------------
	// effect format:
	// attribute count
	// attributes (id,value) pair
	//--------------------------------------------------------------------------

	// read attribute count
	ushort attr_count = SerializerIO_ReadUnsigned(stream);

	// read attributes
	SIValue values[attr_count];
	AttributeID ids[attr_count];

	for(ushort i = 0; i < attr_count; i++) {
		// read attribute ID
		ids[i] = SerializerIO_ReadUnsigned(stream);

		// read attribute value
		values[i] = SIValue_FromBinary(stream);
	}

	AttributeSet attr_set = NULL;
	AttributeSet_AddNoClone(&attr_set, ids, values, attr_count, false);

	return attr_set;
}

static void ApplyCreateNode_V2
(
	SerializerIO stream,  // effects stream
	GraphContext *gc      // graph to operate on
) {
	//--------------------------------------------------------------------------
	// effect format:
	// label count
	// labels
	// attribute count
	// attributes (id,value) pair
	//--------------------------------------------------------------------------

	// read label count
	ushort lbl_count = SerializerIO_ReadUnsigned(stream);

	// read labels
	LabelID labels[lbl_count];
	for(ushort i = 0; i < lbl_count; i++) {
		labels[i] = SerializerIO_ReadUnsigned(stream);
	}

	// read attributes
	AttributeSet attr_set = ReadAttributeSet(stream);

	// create node
	Node n = GE_NEW_NODE();
	CreateNode(gc, &n, labels, lbl_count, attr_set, false);
}

static void ApplyCreateEdge_V2
(
	SerializerIO stream,  // effects stream
	GraphContext *gc      // graph to operate on
) {
	//--------------------------------------------------------------------------
	// effect format:
	// effect type
	// relationship count
	// relationships
	// src node ID
	// dest node ID
	// attribute count
	// attributes (id,value) pair
	//--------------------------------------------------------------------------

	// read relationship type count
	ushort rel_count = SerializerIO_ReadUnsigned(stream);
	ASSERT(rel_count == 1);

	// read relationship type
	RelationID r = SerializerIO_ReadUnsigned(stream);

	// read src node ID
	NodeID src_id = SerializerIO_ReadUnsigned(stream);

	// read dest node ID

	NodeID dest_id = SerializerIO_ReadUnsigned(stream);

	// read attributes
	AttributeSet attr_set = ReadAttributeSet(stream);

	// create edge
	Edge e;
	CreateEdge(gc, &e, src_id, dest_id, r, attr_set, false);
}

// process DeleteNode effect
static void ApplyDeleteNode_V2
(
	SerializerIO stream,  // effects stream
	GraphContext *gc      // graph to operate on
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    node ID
	//--------------------------------------------------------------------------

	Node n;            // node to delete
	Graph *g = gc->g;  // graph to delete node from

	// read node ID off of stream
	EntityID id = SerializerIO_ReadUnsigned(stream);

	// retrieve node from graph
	int res = Graph_GetNode(g, id, &n);
	ASSERT(res != 0);

	// delete node
	DeleteNodes(gc, &n, 1, false);
}

// process DeleteNode effect
static void ApplyDeleteEdge_V2
(
	SerializerIO stream,  // effects stream
	GraphContext *gc      // graph to operate on
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    edge ID
	//    relation ID
	//    src ID
	//    dest ID
	//--------------------------------------------------------------------------

	Graph *g = gc->g;  // graph to delete edge from

	// read edge ID
	EdgeID id = SerializerIO_ReadUnsigned(stream);

	// read relation ID
	RelationID r_id = SerializerIO_ReadUnsigned(stream);

	// read src node ID
	NodeID s_id = SerializerIO_ReadUnsigned(stream);

	// read dest node ID
	NodeID t_id = SerializerIO_ReadUnsigned(stream);

	// get edge from the graph
	Edge e;
	int res = Graph_GetEdge(g, id, (Edge*)&e);
	ASSERT(res != 0);

	// set edge relation, src and destination node
	Edge_SetSrcNodeID (&e, s_id);
	Edge_SetDestNodeID(&e, t_id);
	Edge_SetRelationID(&e, r_id);

	// delete edge
	DeleteEdges(gc, &e, 1, false);
}

static void ApplyLabels_V2
(
	SerializerIO stream,  // effects stream
	GraphContext *gc,     // graph to operate on
	bool add              // add or remove labels
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    node ID
	//    labels count
	//    label IDs
	//--------------------------------------------------------------------------
	
	// read node ID
	EntityID id = SerializerIO_ReadUnsigned(stream);

	// get updated node
	Node  n;
	Graph *g = gc->g;

	bool found = Graph_GetNode(g, id, &n);
	ASSERT(found == true);

	// read labels count
	uint8_t lbl_count = SerializerIO_ReadUnsigned(stream);
	ASSERT(lbl_count > 0);

	// TODO: move to LabelID
	uint n_add_labels          = 0;
	uint n_remove_labels       = 0;
	const char **add_labels    = NULL;
	const char **remove_labels = NULL;
	const char *lbl[lbl_count];

	// assign lbl to the appropriate array
	if(add) {
		add_labels   = lbl;
		n_add_labels = lbl_count;
	} else {
		remove_labels   = lbl;
		n_remove_labels = lbl_count;
	}

	//--------------------------------------------------------------------------
	// read labels
	//--------------------------------------------------------------------------

	for(ushort i = 0; i < lbl_count; i++) {
		LabelID l = SerializerIO_ReadUnsigned(stream);
		Schema *s = GraphContext_GetSchemaByID(gc, l, SCHEMA_NODE);
		ASSERT(s != NULL);
		lbl[i] = Schema_GetName(s);
	}

	//--------------------------------------------------------------------------
	// update node labels
	//--------------------------------------------------------------------------

	UpdateNodeLabels(gc, &n, add_labels, remove_labels, n_add_labels,
			n_remove_labels, false);
}

static void ApplyAddSchema_V2
(
	SerializerIO stream,  // effects stream
	GraphContext *gc      // graph to operate on
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    schema type
	//    schema name
	//--------------------------------------------------------------------------

	// read schema type
	SchemaType t = SerializerIO_ReadUnsigned(stream);

	// read schema name
	char *schema_name = SerializerIO_ReadBuffer(stream, NULL);

	// create schema
	AddSchema(gc, schema_name, t, false);
	rm_free(schema_name);
}

static void ApplyAddAttribute_V2
(
	SerializerIO stream,  // effects stream
	GraphContext *gc      // graph to operate on
) {
	//--------------------------------------------------------------------------
	// effect format:
	// effect type
	// attribute name
	//--------------------------------------------------------------------------
	
	// read attribute name
	char *attr = SerializerIO_ReadBuffer(stream, NULL);

	// attr should not exist
	ASSERT(GraphContext_GetAttributeID(gc, attr) == ATTRIBUTE_ID_NONE);

	// add attribute
	FindOrAddAttribute(gc, attr, false);
	rm_free(attr);
}

// process Update_Edge effect
static void ApplyUpdateEdge_V2
(
	SerializerIO stream,  // effects stream
	GraphContext *gc      // graph to operate on
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    edge ID
	//    relation ID
	//    src node ID
	//    dest node ID
	//    path
	//    attribute value
	//--------------------------------------------------------------------------
	
	uint props_set;       // number of attributes updated
	uint props_removed;   // number of attributes removed
	uint8_t     n;        // sub_path length

	// read edge ID
	EntityID id = SerializerIO_ReadUnsigned(stream);
	ASSERT(id != INVALID_ENTITY_ID);

	// read relation ID
	RelationID r_id = SerializerIO_ReadUnsigned(stream);
	ASSERT(r_id >= 0);

	// read src ID
	NodeID s_id = SerializerIO_ReadUnsigned(stream);
	ASSERT(s_id != INVALID_ENTITY_ID);

	// read dest ID
	NodeID t_id = SerializerIO_ReadUnsigned(stream);
	ASSERT(t_id != INVALID_ENTITY_ID);

	// read attribute ID
	AttributeID attr_id = SerializerIO_ReadUnsigned(stream);

	//--------------------------------------------------------------------------
	// read path's length
	//--------------------------------------------------------------------------

//	fread_assert(&n, sizeof(uint8_t), stream);
//	const char *path[n];
//
//	for(uint8_t i = 0; i < n; i++) {
//		// read sub path element
//		size_t l;
//		fread_assert(&l, sizeof(l), stream);
//
//		const char *attr = rm_malloc(sizeof(char) * l);
//		fread_assert(attr, l, stream);
//		path[i] = attr;
//	}

	//--------------------------------------------------------------------------
	// read attribute value
	//--------------------------------------------------------------------------

	SIValue v = SIValue_FromBinary(stream);
	ASSERT(SI_TYPE(v) & (SI_VALID_PROPERTY_VALUE | T_NULL));
//	ASSERT((attr_id != ATTRIBUTE_ID_ALL || SIValue_IsNull(v)) && attr_id != ATTRIBUTE_ID_NONE);
//
//	UpdateEdgeProperty(gc, id, r_id, s_id, t_id, attr_id, v);
}

// process UpdateNode effect
static void ApplyUpdateNode_V2
(
    SerializerIO stream,  // stream to read value from
	GraphContext *gc      // graph to operate on
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    entity ID
	//    attribute ID
	//    attribute value
	//--------------------------------------------------------------------------

	SIValue v;            // updated value
	uint props_set;       // number of attributes updated
	uint props_removed;   // number of attributes removed

	// read node ID
	EntityID id = SerializerIO_ReadUnsigned(stream);

	// read attribute ID
	AttributeID attr_id = SerializerIO_ReadUnsigned(stream);

	// read attribute value
	v = SIValue_FromBinary(stream);
	ASSERT(SI_TYPE(v) & (SI_VALID_PROPERTY_VALUE | T_NULL));
	ASSERT((attr_id != ATTRIBUTE_ID_ALL || SIValue_IsNull(v)) && attr_id != ATTRIBUTE_ID_NONE);

	UpdateNodeProperty(gc, id, attr_id, v);
}

// applys effects encoded in stream
void Effects_Apply_V2
(
	GraphContext *gc,     // graph to operate on
	FILE *stream,      // effects stream
	size_t l              // stream length
) {
	// validations
	ASSERT(l      > 0);
	ASSERT(gc     != NULL);
	ASSERT(stream != NULL);

	SerializerIO serializer = SerializerIO_FromStream(stream);

	// as long as there's data in stream
	while(ftell(stream) < l) {
		// read effect type
		EffectType t = ReadEffectType(serializer);
		switch(t) {
			case EFFECT_DELETE_NODE:
				ApplyDeleteNode_V2(serializer, gc);
				break;
			case EFFECT_DELETE_EDGE:
				ApplyDeleteEdge_V2(serializer, gc);
				break;
			case EFFECT_UPDATE_NODE:
				ApplyUpdateNode_V2(serializer, gc);
				break;
			case EFFECT_UPDATE_EDGE:
				ApplyUpdateEdge_V2(serializer, gc);
				break;
			case EFFECT_CREATE_NODE:    
				ApplyCreateNode_V2(serializer, gc);
				break;
			case EFFECT_CREATE_EDGE:
				ApplyCreateEdge_V2(serializer, gc);
				break;
			case EFFECT_SET_LABELS:
				ApplyLabels_V2(serializer, gc, true);
				break;
			case EFFECT_REMOVE_LABELS: 
				ApplyLabels_V2(serializer, gc, false);
				break;
			case EFFECT_ADD_SCHEMA:
				ApplyAddSchema_V2(serializer, gc);
				break;
			case EFFECT_ADD_ATTRIBUTE:
				ApplyAddAttribute_V2(serializer, gc);
				break;
			default:
				assert(false && "unknown effect type");
				break;
		}
	}

	SerializerIO_Free(&serializer);
}

