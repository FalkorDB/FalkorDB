/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../../effects.h"
#include "../../../../graph/graph_hub.h"

#include <stdio.h>

// read effect type from stream
static inline EffectType ReadEffectType
(
	FILE *stream  // effects stream
) {
	EffectType t = EFFECT_UNKNOWN;  // default to unknown effect type

	// read EffectType off of stream
	fread_assert(&t, sizeof(EffectType), stream);

	return t;
}

static AttributeSet ReadAttributeSet
(
	FILE *stream
) {
	//--------------------------------------------------------------------------
	// effect format:
	// attribute count
	// attributes (id,value) pair
	//--------------------------------------------------------------------------

	//--------------------------------------------------------------------------
	// read attribute count
	//--------------------------------------------------------------------------

	ushort attr_count;
	fread_assert(&attr_count, sizeof(attr_count), stream);

	//--------------------------------------------------------------------------
	// read attributes
	//--------------------------------------------------------------------------

	SIValue values[attr_count];
	AttributeID ids[attr_count];

	// wrap stream with serializer, required for SIValue decoding
	SerializerIO serializer = SerializerIO_FromStream(stream);
	for(ushort i = 0; i < attr_count; i++) {
		// read attribute ID
		fread_assert(ids + i, sizeof(AttributeID), stream);
		
		// read attribute value
		values[i] = SIValue_FromBinary(serializer);
	}
	SerializerIO_Free(&serializer);

	AttributeSet attr_set = NULL;
	AttributeSet_AddNoClone(&attr_set, ids, values, attr_count, false);

	return attr_set;
}

static void ApplyCreateNode_V1
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
) {
	//--------------------------------------------------------------------------
	// effect format:
	// label count
	// labels
	// attribute count
	// attributes (id,value) pair
	//--------------------------------------------------------------------------

	//--------------------------------------------------------------------------
	// read label count
	//--------------------------------------------------------------------------

	ushort lbl_count;
	fread_assert(&lbl_count, sizeof(lbl_count), stream);

	//--------------------------------------------------------------------------
	// read labels
	//--------------------------------------------------------------------------

	LabelID labels[lbl_count];
	for(ushort i = 0; i < lbl_count; i++) {
		fread_assert(labels + i, sizeof(LabelID), stream);
	}

	//--------------------------------------------------------------------------
	// read attributes
	//--------------------------------------------------------------------------

	AttributeSet attr_set = ReadAttributeSet(stream);

	//--------------------------------------------------------------------------
	// create node
	//--------------------------------------------------------------------------

	Node n = GE_NEW_NODE();
	CreateNode(gc, &n, labels, lbl_count, attr_set, false);
}

static void ApplyCreateEdge_V1
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
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

	//--------------------------------------------------------------------------
	// read relationship type count
	//--------------------------------------------------------------------------

	ushort rel_count;
	fread_assert(&rel_count, sizeof(rel_count), stream);
	ASSERT(rel_count == 1);

	//--------------------------------------------------------------------------
	// read relationship type
	//--------------------------------------------------------------------------

	RelationID r;
	fread_assert(&r, sizeof(r), stream);

	//--------------------------------------------------------------------------
	// read src node ID
	//--------------------------------------------------------------------------

	NodeID src_id;
	fread_assert(&src_id, sizeof(NodeID), stream);

	//--------------------------------------------------------------------------
	// read dest node ID
	//--------------------------------------------------------------------------

	NodeID dest_id;
	fread_assert(&dest_id, sizeof(NodeID), stream);

	//--------------------------------------------------------------------------
	// read attributes
	//--------------------------------------------------------------------------

	AttributeSet attr_set = ReadAttributeSet(stream);

	//--------------------------------------------------------------------------
	// create edge
	//--------------------------------------------------------------------------

	Edge e;
	CreateEdge(gc, &e, src_id, dest_id, r, attr_set, false);
}

static void ApplyLabels_V1
(
	FILE *stream,     // effects stream
	GraphContext *gc, // graph to operate on
	bool add          // add or remove labels
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    node ID
	//    labels count
	//    label IDs
	//--------------------------------------------------------------------------
	
	//--------------------------------------------------------------------------
	// read node ID
	//--------------------------------------------------------------------------

	EntityID id;
	fread_assert(&id, sizeof(id), stream);

	//--------------------------------------------------------------------------
	// get updated node
	//--------------------------------------------------------------------------

	Node  n;
	Graph *g = gc->g;

	bool found = Graph_GetNode(g, id, &n);
	ASSERT(found == true);

	//--------------------------------------------------------------------------
	// read labels count
	//--------------------------------------------------------------------------

	uint8_t lbl_count;
	fread_assert(&lbl_count, sizeof(lbl_count), stream);
	ASSERT(lbl_count > 0);

	// TODO: move to LabelID
	uint n_add_labels          = 0;
	uint n_remove_labels       = 0;
	const char **add_labels    = NULL;
	const char **remove_labels = NULL;
	const char *lbl[lbl_count];

	// assign lbl to the appropriate array
	if(add) {
		add_labels = lbl;
		n_add_labels = lbl_count;
	} else {
		remove_labels = lbl;
		n_remove_labels = lbl_count;
	}

	//--------------------------------------------------------------------------
	// read labels
	//--------------------------------------------------------------------------

	for(ushort i = 0; i < lbl_count; i++) {
		LabelID l;
		fread_assert(&l, sizeof(LabelID), stream);
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

static void ApplyAddSchema_V1
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    schema type
	//    schema name
	//--------------------------------------------------------------------------

	// read schema type
	SchemaType t;
	fread_assert(&t, sizeof(t), stream);

	// read schema name
	// read string length
	size_t l;
	fread_assert(&l, sizeof(l), stream);

	// read string
	char schema_name[l];
	fread_assert(schema_name, l, stream);

	// create schema
	AddSchema(gc, schema_name, t, false);
}

static void ApplyAddAttribute_V1
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
) {
	//--------------------------------------------------------------------------
	// effect format:
	// effect type
	// attribute name
	//--------------------------------------------------------------------------
	
	// read attribute name length
	size_t l;
	fread_assert(&l, sizeof(l), stream);

	// read attribute name
	const char attr[l];
	fread_assert(attr, l, stream);

	// attr should not exist
	ASSERT(GraphContext_GetAttributeID(gc, attr) == ATTRIBUTE_ID_NONE);

	// add attribute
	FindOrAddAttribute(gc, attr, false);
}

// process Update_Edge effect
static void ApplyUpdateEdge_V1
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    edge ID
	//    relation ID
	//    src node ID
	//    dest node ID
	//    attribute ID
	//    attribute value
	//--------------------------------------------------------------------------
	
	SIValue v;            // updated value
	uint props_set;       // number of attributes updated
	uint props_removed;   // number of attributes removed
	AttributeID attr_id;  // entity ID

	NodeID     s_id = INVALID_ENTITY_ID;       // edge src node ID
	NodeID     t_id = INVALID_ENTITY_ID;       // edge dest node ID
	RelationID r_id = GRAPH_UNKNOWN_RELATION;  // edge rel-type

	EntityID id = INVALID_ENTITY_ID;

	//--------------------------------------------------------------------------
	// read edge ID
	//--------------------------------------------------------------------------

	fread_assert(&id, sizeof(EntityID), stream);
	ASSERT(id != INVALID_ENTITY_ID);

	//--------------------------------------------------------------------------
	// read relation ID
	//--------------------------------------------------------------------------

	fread_assert(&r_id, sizeof(RelationID), stream);
	ASSERT(r_id >= 0);

	//--------------------------------------------------------------------------
	// read src ID
	//--------------------------------------------------------------------------

	fread_assert(&s_id, sizeof(NodeID), stream);
	ASSERT(s_id != INVALID_ENTITY_ID);

	//--------------------------------------------------------------------------
	// read dest ID
	//--------------------------------------------------------------------------

	fread_assert(&t_id, sizeof(NodeID), stream);
	ASSERT(t_id != INVALID_ENTITY_ID);

	//--------------------------------------------------------------------------
	// read attribute ID
	//--------------------------------------------------------------------------

	fread_assert(&attr_id, sizeof(AttributeID), stream);

	//--------------------------------------------------------------------------
	// read attribute value
	//--------------------------------------------------------------------------

	// wrap stream with serializer, required for SIValue decoding
	SerializerIO serializer = SerializerIO_FromStream(stream);
	v = SIValue_FromBinary(serializer);
	SerializerIO_Free(&serializer);

	ASSERT(SI_TYPE(v) & (SI_VALID_PROPERTY_VALUE | T_NULL));
	ASSERT((attr_id != ATTRIBUTE_ID_ALL || SIValue_IsNull(v)) && attr_id != ATTRIBUTE_ID_NONE);

	UpdateEdgeProperty(gc, id, r_id, s_id, t_id, attr_id, v);
}

// process UpdateNode effect
static void ApplyUpdateNode_V1
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
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
	AttributeID attr_id;  // entity ID

	EntityID id = INVALID_ENTITY_ID;

	//--------------------------------------------------------------------------
	// read node ID
	//--------------------------------------------------------------------------

	fread_assert(&id, sizeof(EntityID), stream);

	//--------------------------------------------------------------------------
	// read attribute ID
	//--------------------------------------------------------------------------

	fread_assert(&attr_id, sizeof(AttributeID), stream);

	//--------------------------------------------------------------------------
	// read attribute ID
	//--------------------------------------------------------------------------

	// wrap stream with serializer, required for SIValue decoding
	SerializerIO serializer = SerializerIO_FromStream(stream);
	v = SIValue_FromBinary(serializer);
	SerializerIO_Free(&serializer);

	ASSERT(SI_TYPE(v) & (SI_VALID_PROPERTY_VALUE | T_NULL));
	ASSERT((attr_id != ATTRIBUTE_ID_ALL || SIValue_IsNull(v)) && attr_id != ATTRIBUTE_ID_NONE);

	UpdateNodeProperty(gc, id, attr_id, v);
}

// process DeleteNode effect
static void ApplyDeleteNode_V1
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    node ID
	//--------------------------------------------------------------------------
	
	Node n;            // node to delete
	EntityID id;       // node ID
	Graph *g = gc->g;  // graph to delete node from

	// read node ID off of stream
	fread_assert(&id, sizeof(EntityID), stream);

	// retrieve node from graph
	int res = Graph_GetNode(g, id, &n);
	ASSERT(res != 0);

	// delete node
	DeleteNodes(gc, &n, 1, false);
}

// process DeleteNode effect
static void ApplyDeleteEdge_V1
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    edge ID
	//    relation ID
	//    src ID
	//    dest ID
	//--------------------------------------------------------------------------

	Edge e;  // edge to delete

	EntityID id   = INVALID_ENTITY_ID;       // edge ID
	int      r_id = GRAPH_UNKNOWN_RELATION;  // edge rel-type
	NodeID   s_id = INVALID_ENTITY_ID;       // edge src node ID
	NodeID   t_id = INVALID_ENTITY_ID;       // edge dest node ID

	int res;
	UNUSED(res);

	Graph *g = gc->g;  // graph to delete edge from

	// read edge ID
	fread_assert(&id, sizeof(EntityID), stream);

	// read relation ID
	fread_assert(&r_id, sizeof(RelationID), stream);

	// read src node ID
	fread_assert(&s_id, sizeof(EntityID), stream);

	// read dest node ID
	fread_assert(&t_id, sizeof(EntityID), stream);

	// get edge from the graph
	res = Graph_GetEdge(g, id, (Edge*)&e);
	ASSERT(res != 0);

	// set edge relation, src and destination node
	Edge_SetSrcNodeID(&e, s_id);
	Edge_SetDestNodeID(&e, t_id);
	Edge_SetRelationID(&e, r_id);

	// delete edge
	DeleteEdges(gc, &e, 1, false);
}

// applys effects encoded in stream
void Effects_Apply_V1
(
	GraphContext *gc,  // graph to operate on
	FILE *stream,      // effects stream
	size_t l           // stream length
) {
	// validations
	ASSERT(l      > 0);      // stream can't be empty
	ASSERT(stream != NULL);  // stream can't be NULL

	// as long as there's data in stream
	while(ftell(stream) < l) {
		// read effect type
		EffectType t = ReadEffectType(stream);
		switch(t) {
			case EFFECT_DELETE_NODE:
				ApplyDeleteNode_V1(stream, gc);
				break;
			case EFFECT_DELETE_EDGE:
				ApplyDeleteEdge_V1(stream, gc);
				break;
			case EFFECT_UPDATE_NODE:
				ApplyUpdateNode_V1(stream, gc);
				break;
			case EFFECT_UPDATE_EDGE:
				ApplyUpdateEdge_V1(stream, gc);
				break;
			case EFFECT_CREATE_NODE:    
				ApplyCreateNode_V1(stream, gc);
				break;
			case EFFECT_CREATE_EDGE:
				ApplyCreateEdge_V1(stream, gc);
				break;
			case EFFECT_SET_LABELS:
				ApplyLabels_V1(stream, gc, true);
				break;
			case EFFECT_REMOVE_LABELS: 
				ApplyLabels_V1(stream, gc, false);
				break;
			case EFFECT_ADD_SCHEMA:
				ApplyAddSchema_V1(stream, gc);
				break;
			case EFFECT_ADD_ATTRIBUTE:
				ApplyAddAttribute_V1(stream, gc);
				break;
			default:
				assert(false && "unknown effect type");
				break;
		}
	}
}

