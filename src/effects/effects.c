/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects.h"
#include "../query_ctx.h"
#include "../datatypes/map.h"
#include "../datatypes/vector.h"

// effects buffer is a linked-list of buffers
struct _EffectsBuffer {
	uint64_t n;               // number of effects encoded
	FILE *stream;             // effects get written into this stream
	char *buffer;             // effects stream buffer
	size_t buffer_size;       // effects stream buffer size
	SerializerIO serializer;  // effects encoder
};

// create a new effects-buffer
EffectsBuffer *EffectsBuffer_New(void) {
	EffectsBuffer *eb = rm_malloc(sizeof(EffectsBuffer));

	// init effects buffer
	eb->n           = 0;
	eb->buffer      = NULL;
	eb->buffer_size = 0;

	// create memory stream
	eb->stream = open_memstream(&eb->buffer, &eb->buffer_size);

	// create encoder
	eb->serializer = SerializerIO_FromStream(eb->stream);

	// write effects version to newly created buffer
	uint8_t v = EFFECTS_VERSION;
	SerializerIO_WriteUnsigned(eb->serializer, v);

	return eb;
}

// reset effects-buffer
void EffectsBuffer_Reset
(
	EffectsBuffer *eb  // effects-buffer
) {
	ASSERT(eb != NULL);

	// reset effects count
	eb->n = 0;

	// seek to the begining of the stream
	rewind(eb->stream);

	// write effects version
	SerializerIO_WriteUnsigned(eb->serializer, EFFECTS_VERSION);
}

// increase effects count
static inline void EffectsBuffer_IncEffectCount
(
	EffectsBuffer *eb  // effects buffer
) {
	ASSERT(eb != NULL);
	
	eb->n++;
}

// dump attributes into stream
static void _WriteAttributeSet
(
	EffectsBuffer *eb,        // effects buffer
	const AttributeSet attrs  // attribute set to write to stream
) {
	ASSERT(eb    != NULL);
	ASSERT(attrs != NULL);

	//--------------------------------------------------------------------------
	// write attribute count
	//--------------------------------------------------------------------------

	ushort attr_count = AttributeSet_Count(attrs);
	SerializerIO_WriteUnsigned(eb->serializer, attr_count);

	//--------------------------------------------------------------------------
	// write attributes
	//--------------------------------------------------------------------------

	for(ushort i = 0; i < attr_count; i++) {
		// get current attribute id and value
		AttributeID attr_id;
		SIValue attr = AttributeSet_GetIdx(attrs, i, &attr_id);

		// write attribute ID
		SerializerIO_WriteUnsigned(eb->serializer, attr_id);

		// write attribute value
		SIValue_ToBinary(eb->serializer, &attr);
	}
}

// returns number of effects in buffer
uint64_t EffectsBuffer_Length
(
	const EffectsBuffer *eb  // effects-buffer
) {
	ASSERT(eb != NULL);
	
	return eb->n;
}

// gets the effects-buffer internal buffer
char *EffectsBuffer_Buffer
(
	EffectsBuffer *eb,  // effects-buffer
	size_t *n           // [output] size of returned buffer
) {
	ASSERT(n  != NULL);
	ASSERT(eb != NULL);

	// flush stream
	fflush(eb->stream);

	// set buffer size
	*n = eb->buffer_size;
	return eb->buffer;
}

//------------------------------------------------------------------------------
// effects creation API
//------------------------------------------------------------------------------

// add a node creation effect to buffer
void EffectsBuffer_AddCreateNodeEffect
(
	EffectsBuffer *eb,      // effects buffer
	const Node *n,          // node created
	const LabelID *labels,  // node labels
	ushort label_count      // number of labels
) {
	ASSERT(n  != NULL);
	ASSERT(eb != NULL);

	//--------------------------------------------------------------------------
	// effect format:
	// effect type
	// label count
	// labels
	// attribute count
	// attributes (id, value) pair
	//--------------------------------------------------------------------------
	
	// TODO: find a better place for this logic
	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics();
	stats->nodes_created++;
	stats->properties_set += AttributeSet_Count(*n->attributes);

	// write effect type
	EffectType t = EFFECT_CREATE_NODE;
	SerializerIO_WriteUnsigned(eb->serializer, t);

	// write label count
	SerializerIO_WriteUnsigned(eb->serializer, label_count);

	// write labels
	for(ushort i = 0; i < label_count; i++) {
		SerializerIO_WriteUnsigned(eb->serializer, labels[i]);
	}

	// write attribute set
	const AttributeSet attrs = GraphEntity_GetAttributes((const GraphEntity*)n);
	_WriteAttributeSet(eb, attrs);

	EffectsBuffer_IncEffectCount(eb);
}

// add a edge creation effect to buffer
void EffectsBuffer_AddCreateEdgeEffect
(
	EffectsBuffer *eb,  // effects buffer
	const Edge *edge    // edge created
) {
	ASSERT(eb   != NULL);
	ASSERT(edge != NULL);

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
	
	// TODO: find a better place for this logic
	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics();
	stats->relationships_created++;
	stats->properties_set += AttributeSet_Count(*edge->attributes);

	// write effect type
	EffectType t = EFFECT_CREATE_EDGE;
	SerializerIO_WriteUnsigned(eb->serializer, t);

	// write relationship type
	ushort rel_count = 1;
	SerializerIO_WriteUnsigned(eb->serializer, rel_count);

	RelationID rel_id = Edge_GetRelationID(edge);
	SerializerIO_WriteUnsigned(eb->serializer, rel_id);

	// write src node ID
	NodeID src_id = Edge_GetSrcNodeID(edge);
	SerializerIO_WriteUnsigned(eb->serializer, src_id);

	// write dest node ID
	NodeID dest_id = Edge_GetDestNodeID(edge);
	SerializerIO_WriteUnsigned(eb->serializer, dest_id);

	// write attribute set 
	const AttributeSet attrs = GraphEntity_GetAttributes((GraphEntity*)edge);
	_WriteAttributeSet(eb, attrs);

	EffectsBuffer_IncEffectCount(eb);
}

// add a node deletion effect to buffer
void EffectsBuffer_AddDeleteNodeEffect
(
	EffectsBuffer *eb,  // effects buffer
	const Node *node    // node deleted
) {
	ASSERT(eb   != NULL);
	ASSERT(node != NULL);

	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    node ID
	//--------------------------------------------------------------------------

	// TODO: find a better place for this logic
	QueryCtx_GetResultSetStatistics()->nodes_deleted++;

	// write effect type
	EffectType t = EFFECT_DELETE_NODE;
	SerializerIO_WriteUnsigned(eb->serializer, t);

	// write node ID
	SerializerIO_WriteUnsigned(eb->serializer, ENTITY_GET_ID(node));

	EffectsBuffer_IncEffectCount(eb);
}

// add a edge deletion effect to buffer
void EffectsBuffer_AddDeleteEdgeEffect
(
	EffectsBuffer *eb,  // effects buffer
	const Edge *edge    // edge deleted
) {
	ASSERT(eb   != NULL);
	ASSERT(edge != NULL);

	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    edge ID
	//    relation ID
	//    src ID
	//    dest ID
	//--------------------------------------------------------------------------

	// TODO: find a better place for this logic
	QueryCtx_GetResultSetStatistics()->relationships_deleted++;

	// write effect type
	EffectType t = EFFECT_DELETE_EDGE;
	SerializerIO_WriteUnsigned(eb->serializer, t);

	// write edge ID
	SerializerIO_WriteUnsigned(eb->serializer, ENTITY_GET_ID(edge));

	// write edge relation
	RelationID r_id = Edge_GetRelationID(edge);
	SerializerIO_WriteUnsigned(eb->serializer, r_id);

	// write edge source node ID
	NodeID src_id = Edge_GetSrcNodeID(edge);
	SerializerIO_WriteUnsigned(eb->serializer, src_id);

	// write edge destination node ID
	NodeID dest_id = Edge_GetDestNodeID(edge);
	SerializerIO_WriteUnsigned(eb->serializer, dest_id);

	EffectsBuffer_IncEffectCount(eb);
};

// add an entity update effect to buffer
static void EffectsBuffer_AddNodeUpdateEffect
(
	EffectsBuffer *eb,    // effects buffer
	Node *node,           // updated node
	AttributeID attr_id,  // updated attribute ID
	const char **path,    // sub path
	uint8_t n,            // sub path length
 	SIValue value         // value
) {
	ASSERT(eb   != NULL);
	ASSERT(node != NULL);
	ASSERT(SI_TYPE(value) & SI_VALID_PROPERTY_VALUE);

	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    entity ID
	//    attribute id
	//    path's length
	//    path
	//    attribute value
	//--------------------------------------------------------------------------

	// write effect type
	EffectType t = EFFECT_UPDATE_NODE;
	SerializerIO_WriteUnsigned(eb->serializer, t);

	// write entity ID
	SerializerIO_WriteUnsigned(eb->serializer, ENTITY_GET_ID(node));

	// write attribute ID
	SerializerIO_WriteUnsigned(eb->serializer, attr_id);

	// write sub path length
	//EffectsBuffer_WriteBytes(&n, sizeof(uint8_t), buff);

	// write sub path
	//for(uint8_t i = 0; i < n; i++) {
	//	EffectsBuffer_WriteString(path[i], buff);
	//}

	// write attribute value
	SIValue_ToBinary(eb->serializer, &value);

	EffectsBuffer_IncEffectCount(eb);
}

// add an entity update effect to buffer
static void EffectsBuffer_AddEdgeUpdateEffect
(
	EffectsBuffer *eb,    // effects buffer
	Edge *edge,           // updated edge
	AttributeID attr_id,  // updated attribute ID
	const char **path,    // sub path
	uint8_t n,            // sub path length
 	SIValue value         // value
) {
	ASSERT(eb    != NULL);
	ASSERT(edge  != NULL);
	ASSERT(SI_TYPE(value) & SI_VALID_PROPERTY_VALUE);

	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    edge ID
	//    relation ID
	//    src ID
	//    dest ID
	//    attribute id
	//    path's length
	//    path
	//    value
	//--------------------------------------------------------------------------

	// write effect type
	EffectType t = EFFECT_UPDATE_EDGE;
	SerializerIO_WriteUnsigned(eb->serializer, t);

	// write edge ID
	SerializerIO_WriteUnsigned(eb->serializer, ENTITY_GET_ID(edge));

	// write relation ID
	RelationID r = Edge_GetRelationID(edge);
	SerializerIO_WriteUnsigned(eb->serializer, r);

	// write src ID
	NodeID s = Edge_GetSrcNodeID(edge);
	SerializerIO_WriteUnsigned(eb->serializer, s);

	// write dest ID
	NodeID d = Edge_GetDestNodeID(edge);
	SerializerIO_WriteUnsigned(eb->serializer, d);

	// write attribute ID
	SerializerIO_WriteUnsigned(eb->serializer, attr_id);

	// write sub path length
	//EffectsBuffer_WriteBytes(&n, sizeof(uint8_t), buff);

	// write sub path
	//for(uint8_t i = 0; i < n; i++) {
	//	EffectsBuffer_WriteString(path[i], buff);
	//}

	// write attribute value
	SIValue_ToBinary(eb->serializer, &value);

	EffectsBuffer_IncEffectCount(eb);
}

// add an entity attribute removal effect to buffer
void EffectsBuffer_AddEntityRemoveAttributeEffect
(
	EffectsBuffer *eb,           // effects buffer
	GraphEntity *entity,         // updated entity ID
	AttributeID attr_id,         // updated attribute ID
	const char **path,           // sub path
	uint8_t l,                   // sub path length
	GraphEntityType entity_type  // entity type
) {
	ASSERT(eb     != NULL);
	ASSERT(entity != NULL);

	// attribute was deleted
	int n = (attr_id == ATTRIBUTE_ID_ALL)
		? AttributeSet_Count(*entity->attributes)
		: 1;

	// TODO: find a better place for this logic
	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics();
	stats->properties_removed += n;

	SIValue v = SI_NullVal();
	if(entity_type == GETYPE_NODE) {
		EffectsBuffer_AddNodeUpdateEffect(eb, (Node*)entity, attr_id, path,
				n, v);
	} else {
		EffectsBuffer_AddEdgeUpdateEffect(eb, (Edge*)entity, attr_id, path,
				n, v);
	}
}

// add an entity add new attribute effect to buffer
void EffectsBuffer_AddEntityAddAttributeEffect
(
	EffectsBuffer *eb,           // effects buffer
	GraphEntity *entity,         // updated entity ID
	AttributeID attr_id,         // updated attribute ID
	const char **path,           // sub path
	uint8_t n,                   // sub path length
	SIValue value,               // value
	GraphEntityType entity_type  // entity type
) {
	ASSERT(eb     != NULL);
	ASSERT(entity != NULL);
	ASSERT(SI_TYPE(value)  & SI_VALID_PROPERTY_VALUE);

	// TODO: find a better place for this logic
	// attribute was added
	QueryCtx_GetResultSetStatistics()->properties_set++;

	if(entity_type == GETYPE_NODE) {
		EffectsBuffer_AddNodeUpdateEffect(eb, (Node*)entity, attr_id, path,
				n, value);
	} else {
		EffectsBuffer_AddEdgeUpdateEffect(eb, (Edge*)entity, attr_id, path,
				n, value);
	}
}

// add an entity update attribute effect to buffer
void EffectsBuffer_AddEntityUpdateAttributeEffect
(
	EffectsBuffer *eb,           // effects buffer
	GraphEntity *entity,         // updated entity ID
	AttributeID attr_id,         // updated attribute ID
	const char **path,           // sub path
	uint8_t n,                   // sub path length
	SIValue value,               // value
	GraphEntityType entity_type  // entity type
) {
	ASSERT(eb     != NULL);
	ASSERT(entity != NULL);
	ASSERT(SI_TYPE(value)  & SI_VALID_PROPERTY_VALUE);

	// TODO: find a better place for this logic
	ResultSetStatistics *stats = QueryCtx_GetResultSetStatistics();
	stats->properties_set++; // attribute was set
	stats->properties_removed++; // old attribute was deleted

	if(entity_type == GETYPE_NODE) {
		EffectsBuffer_AddNodeUpdateEffect(eb, (Node*)entity, attr_id, path, n,
				value);
	} else {
		EffectsBuffer_AddEdgeUpdateEffect(eb, (Edge*)entity, attr_id, path, n,
				value);
	}
}

// add a node add label effect to buffer
void EffectsBuffer_AddSetRemoveLabelsEffect
(
	EffectsBuffer *eb,       // effects buffer
	const Node *node,        // updated node
	const LabelID *lbl_ids,  // labels
	uint8_t lbl_count,       // number of labels
	EffectType t             // effect type
) {
	ASSERT(eb        != NULL);
	ASSERT(node      != NULL);
	ASSERT(lbl_ids   != NULL);
	ASSERT(lbl_count > 0);

	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    node ID
	//    labels count
	//    label IDs
	//--------------------------------------------------------------------------

	// write effect type
	SerializerIO_WriteUnsigned(eb->serializer, t);

	// write node ID
	SerializerIO_WriteUnsigned(eb->serializer, ENTITY_GET_ID(node));
	
	// write labels count
	SerializerIO_WriteUnsigned(eb->serializer, lbl_count);
	
	// write label IDs
	for(ushort i = 0; i < lbl_count; i++) {
		SerializerIO_WriteUnsigned(eb->serializer, lbl_ids[i]);
	}

	EffectsBuffer_IncEffectCount(eb);
}

// add a node add labels effect to buffer
void EffectsBuffer_AddLabelsEffect
(
	EffectsBuffer *eb,       // effects buffer
	const Node *node,        // updated node
	const LabelID *lbl_ids,  // added labels
	size_t lbl_count         // number of removed labels
) {
	ASSERT(eb        != NULL);
	ASSERT(node      != NULL);
	ASSERT(lbl_ids   != NULL);
	ASSERT(lbl_count > 0);

	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    node ID
	//    labels count
	//    label IDs
	//--------------------------------------------------------------------------

	// TODO: find a better place for this logic
	QueryCtx_GetResultSetStatistics()->labels_added += lbl_count;

	EffectType t = EFFECT_SET_LABELS;
	EffectsBuffer_AddSetRemoveLabelsEffect(eb, node, lbl_ids, lbl_count, t);

	EffectsBuffer_IncEffectCount(eb);
}

// add a node remove labels effect to buffer
void EffectsBuffer_AddRemoveLabelsEffect
(
	EffectsBuffer *eb,       // effects buffer
	const Node *node,        // updated node
	const LabelID *lbl_ids,  // removed labels
	size_t lbl_count         // number of removed labels
) {
	ASSERT(eb        != NULL);
	ASSERT(node      != NULL);
	ASSERT(lbl_ids   != NULL);
	ASSERT(lbl_count > 0);

	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    node ID
	//    labels count
	//    label IDs
	//--------------------------------------------------------------------------

	// TODO: find a better place for this logic
	QueryCtx_GetResultSetStatistics()->labels_removed += lbl_count;

	EffectType t = EFFECT_REMOVE_LABELS;
	EffectsBuffer_AddSetRemoveLabelsEffect(eb, node, lbl_ids, lbl_count, t);

	EffectsBuffer_IncEffectCount(eb);
}

// add a schema addition effect to buffer
void EffectsBuffer_AddNewSchemaEffect
(
	EffectsBuffer *eb,        // effects stream
	const char *schema_name,  // id of the schema
	SchemaType st             // type of the schema
) {
	ASSERT(eb          != NULL);
	ASSERT(schema_name != NULL);

	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    schema type
	//    schema name
	//--------------------------------------------------------------------------

	// write effect type
	EffectType t = EFFECT_ADD_SCHEMA;
	SerializerIO_WriteUnsigned(eb->serializer, t);

	// write schema type
	SerializerIO_WriteUnsigned(eb->serializer, st);

	// write schema name
	SerializerIO_WriteBuffer(eb->serializer, schema_name, strlen(schema_name));

	EffectsBuffer_IncEffectCount(eb);
}

// add an attribute addition effect to buffer
void EffectsBuffer_AddNewAttributeEffect
(
	EffectsBuffer *eb,  // effects stream
	const char *attr    // attribute name
) {
	ASSERT(eb   != NULL);
	ASSERT(attr != NULL);

	//--------------------------------------------------------------------------
	// effect format:
	// effect type
	// attribute name
	//--------------------------------------------------------------------------

	// write effect type
	EffectType t = EFFECT_ADD_ATTRIBUTE;
	SerializerIO_WriteUnsigned(eb->serializer, t);

	// write attribute name
	SerializerIO_WriteBuffer(eb->serializer, attr, strlen(attr));

	EffectsBuffer_IncEffectCount(eb);
}

void EffectsBuffer_Free
(
	EffectsBuffer *eb
) {
	if(eb == NULL) return;

	fclose(eb->stream);  // close stream
	free(eb->buffer);    // free buffer

	rm_free(eb);
}

