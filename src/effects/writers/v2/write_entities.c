/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// the v2 entity writers


#include "../../../RG.h"
#include "../../effects.h"
#include "../../effects_internal.h"
#include "write_v2.h"
#include "../../../query_ctx.h"
#include "../../../graph/graph.h"
#include "../../../graph/graphcontext.h"


// add an entity update effect to buffer
static void EffectsBuffer_AddNodeUpdateEffect
(
	EffectsBuffer *buff,  // effect buffer
	Node *node,           // updated node
	AttributeID attr_id,  // updated attribute ID
 	SIValue value         // value
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    entity ID
	//    attribute id
	//    attribute value
	//--------------------------------------------------------------------------

	#pragma pack(push, 1)
	struct {
		EffectType t ;
		EntityID id;
		AttributeID attr_id ;
	} _update_node_desc;
	#pragma pack(pop)

	_update_node_desc.t       = EFFECT_UPDATE_NODE ;
	_update_node_desc.id      = ENTITY_GET_ID (node) ;
	_update_node_desc.attr_id = attr_id ;

	EffectsBuffer_WriteBytes (&_update_node_desc, sizeof(_update_node_desc),
			buff) ;

	//--------------------------------------------------------------------------
	// write attribute value
	//--------------------------------------------------------------------------

	EffectsBuffer_WriteSIValue (&value, buff) ;

	EffectsBuffer_IncEffectCount (buff) ;
}

// add an entity update effect to buffer
static void EffectsBuffer_AddEdgeUpdateEffect
(
	EffectsBuffer *buff,  // effect buffer
	Edge *edge,           // updated edge
	AttributeID attr_id,  // updated attribute ID
 	SIValue value         // value
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    edge ID
	//    relation ID
	//    src ID
	//    dest ID
	//    attribute count (=n)
	//    attributes (id,value) pair
	//--------------------------------------------------------------------------

	#pragma pack(push, 1)
	struct {
		EffectType t ;
		EntityID id;
		RelationID r;
		NodeID s;
		NodeID d;
		AttributeID attr_id ;
	} _update_edge_desc;
	#pragma pack(pop)

	_update_edge_desc.t       = EFFECT_UPDATE_EDGE ;
	_update_edge_desc.id      = ENTITY_GET_ID      (edge) ;
	_update_edge_desc.r       = Edge_GetRelationID (edge) ;
	_update_edge_desc.s       = Edge_GetSrcNodeID  (edge) ;
	_update_edge_desc.d       = Edge_GetDestNodeID (edge) ;
	_update_edge_desc.attr_id = attr_id ;

	EffectsBuffer_WriteBytes (&_update_edge_desc, sizeof (_update_edge_desc),
			buff) ;

	//--------------------------------------------------------------------------
	// write attribute value
	//--------------------------------------------------------------------------

	EffectsBuffer_WriteSIValue(&value, buff);

	EffectsBuffer_IncEffectCount(buff);
}


void EffectsWriteV2_CreateNode
(
	EffectsBuffer *buff,    // effect buffer
	const Node *n,          // node created
	const LabelID *labels,  // node labels
	ushort label_count      // number of labels
) {
	//--------------------------------------------------------------------------
	// effect format:
	// effect type
	// label count
	// labels
	// attribute count
	// attributes (id,value) pair
	//--------------------------------------------------------------------------

	EffectType t = EFFECT_CREATE_NODE;
	EffectsBuffer_WriteBytes(&t, sizeof(t), buff);

	//--------------------------------------------------------------------------
	// write label count
	//--------------------------------------------------------------------------

	EffectsBuffer_WriteBytes(&label_count, sizeof(label_count), buff);

	//--------------------------------------------------------------------------
	// write labels
	//--------------------------------------------------------------------------

	if(label_count > 0) {
		EffectsBuffer_WriteBytes(labels, sizeof(LabelID) * label_count, buff);
	}

	//--------------------------------------------------------------------------
	// write attribute set
	//--------------------------------------------------------------------------

	const AttributeSet attrs = GraphEntity_GetAttributes((const GraphEntity*)n);
	EffectsBuffer_WriteAttributeSet(attrs, buff);

	EffectsBuffer_IncEffectCount(buff);
}


void EffectsWriteV2_CreateEdge
(
	EffectsBuffer *buff,  // effect buffer
	const Edge *edge      // edge created
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

	// encoded edge struct
	#pragma pack(push, 1)
	struct {
		EffectType t ;
		uint16_t rel_count ;
		RelationID r ;
		NodeID src_id ;
		NodeID dest_id ;
	} _create_edge_desc;
	#pragma pack(pop)

	//--------------------------------------------------------------------------
	// populate & write edge
	//--------------------------------------------------------------------------

	_create_edge_desc.t         = EFFECT_CREATE_EDGE ;
	_create_edge_desc.rel_count = 1 ;
	_create_edge_desc.r         = Edge_GetRelationID (edge) ;
	_create_edge_desc.src_id    = Edge_GetSrcNodeID  (edge) ;
	_create_edge_desc.dest_id   = Edge_GetDestNodeID (edge) ;

	EffectsBuffer_WriteBytes (&_create_edge_desc, sizeof (_create_edge_desc),
			buff) ;

	//--------------------------------------------------------------------------
	// write attribute set 
	//--------------------------------------------------------------------------

	const AttributeSet attrs =
		GraphEntity_GetAttributes ((const GraphEntity*)edge) ;

	EffectsBuffer_WriteAttributeSet (attrs, buff) ;

	EffectsBuffer_IncEffectCount (buff) ;
}


void EffectsWriteV2_DeleteNode
(
	EffectsBuffer *buff,  // effect buffer
	const Node *node      // node deleted
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    node ID
	//--------------------------------------------------------------------------

	#pragma pack(push, 1)
	struct {
		EffectType t;
		EntityID id;
	} _delete_node_desc ;
	#pragma pack(pop)

	_delete_node_desc.t  = EFFECT_DELETE_NODE ;
	_delete_node_desc.id = ENTITY_GET_ID (node) ;

	EffectsBuffer_WriteBytes (&_delete_node_desc, sizeof(_delete_node_desc),
			buff) ;

	EffectsBuffer_IncEffectCount (buff) ;
}


void EffectsWriteV2_DeleteEdge
(
	EffectsBuffer *eb,  // effect buffer
	const Edge *edge    // edge deleted
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    edge ID
	//    relation ID
	//    src ID
	//    dest ID
	//--------------------------------------------------------------------------

	// encoded edge struct
	#pragma pack(push, 1)
	struct {
		EffectType t ;
		EntityID id;
		RelationID r ;
		NodeID src_id ;
		NodeID dest_id ;
	} _delete_edge_desc;
	#pragma pack(pop)

	_delete_edge_desc.t       = EFFECT_DELETE_EDGE ;
	_delete_edge_desc.id      = ENTITY_GET_ID      (edge) ;
	_delete_edge_desc.r       = Edge_GetRelationID (edge) ;
	_delete_edge_desc.src_id  = Edge_GetSrcNodeID  (edge) ;
	_delete_edge_desc.dest_id = Edge_GetDestNodeID (edge) ;

	EffectsBuffer_WriteBytes (&_delete_edge_desc, sizeof(_delete_edge_desc),
			eb) ;

	EffectsBuffer_IncEffectCount(eb);
}


void EffectsWriteV2_UpdateEntity
(
	EffectsBuffer *buff,         // effect buffer
	GraphEntity *entity,         // updated entity
	AttributeID attr_id,         // updated attribute, or ATTRIBUTE_ID_ALL
	SIValue value,               // value; a null is a removal
	GraphEntityType entity_type  // entity type
) {

	if(entity_type == GETYPE_NODE) {
		EffectsBuffer_AddNodeUpdateEffect(buff, (Node*)entity, attr_id, value);
	} else {
		EffectsBuffer_AddEdgeUpdateEffect(buff, (Edge*)entity, attr_id, value);
	}
}


void EffectsWriteV2_Labels
(
	EffectsBuffer *buff,  // effect buffer
	GrB_Vector nodes,     // nodes the label applies to
	EffectType opcode     // SET_LABELS or REMOVE_LABELS
) {

	EffectType t = opcode;
	EffectsBuffer_WriteBytes (&t, sizeof (t), buff) ;

	//--------------------------------------------------------------------------
	// encode vector
	//--------------------------------------------------------------------------

	void *blob ;
	GrB_Index blob_size ;
	GrB_OK (GxB_Vector_serialize (&blob, &blob_size, nodes, NULL)) ;

	EffectsBuffer_WriteBytes (&blob_size, sizeof (blob_size), buff) ;
	EffectsBuffer_WriteBytes (blob, blob_size, buff) ;

	rm_free (blob) ;

	EffectsBuffer_IncEffectCount (buff) ;
}


void EffectsWriteV2_NewSchema
(
	EffectsBuffer *buff,      // effect buffer
	const char *schema_name,  // name of the schema
	SchemaType st             // type of the schema
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    schema type
	//    schema name
	//--------------------------------------------------------------------------

	EffectType t = EFFECT_ADD_SCHEMA;
	EffectsBuffer_WriteBytes(&t, sizeof(t), buff);

	//--------------------------------------------------------------------------
	// write schema type
	//--------------------------------------------------------------------------

	EffectsBuffer_WriteBytes(&st, sizeof(st), buff);

	//--------------------------------------------------------------------------
	// write schema name
	//--------------------------------------------------------------------------

	EffectsBuffer_WriteString(schema_name, buff);

	EffectsBuffer_IncEffectCount(buff);
}


void EffectsWriteV2_NewAttribute
(
	EffectsBuffer *buff,  // effect buffer
	const char *attr      // attribute name
) {
	//--------------------------------------------------------------------------
	// effect format:
	// effect type
	// attribute name
	//--------------------------------------------------------------------------

	EffectType t = EFFECT_ADD_ATTRIBUTE;
	EffectsBuffer_WriteBytes(&t, sizeof(t), buff);

	//--------------------------------------------------------------------------
	// write attribute name
	//--------------------------------------------------------------------------

	EffectsBuffer_WriteString(attr, buff);

	EffectsBuffer_IncEffectCount(buff);
}
