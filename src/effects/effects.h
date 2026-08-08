/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../graph/graphcontext.h"
#include "../index/index_field.h"
#include "../constraint/constraint.h"

#define EFFECTS_VERSION 2  // current effects encoding/decoding version

// EffectsBuffer is an opaque data structure
typedef struct _EffectsBuffer EffectsBuffer;

// types of effects
typedef enum {
	EFFECT_UNKNOWN = 0,    // unknown effect
	EFFECT_UPDATE_NODE,    // node update
	EFFECT_UPDATE_EDGE,    // edge update
	EFFECT_CREATE_NODE,    // node creation
	EFFECT_CREATE_EDGE,    // edge creation
	EFFECT_DELETE_NODE,    // node deletion
	EFFECT_DELETE_EDGE,    // edge deletion
	EFFECT_SET_LABELS,     // set labels
	EFFECT_REMOVE_LABELS,  // remove labels
	EFFECT_ADD_SCHEMA,         // schema addition
	EFFECT_ADD_ATTRIBUTE,      // add attribute
	EFFECT_CREATE_INDEX,       // index field creation
	EFFECT_DROP_INDEX,         // index field deletion
	EFFECT_CREATE_CONSTRAINT,  // constraint creation
	EFFECT_DROP_CONSTRAINT,    // constraint deletion
} EffectType;

//------------------------------------------------------------------------------
// effects API
//------------------------------------------------------------------------------

// applies effects encoded in buffer
// returns false if the replica has diverged from the master and the effects
// could not be applied in full; on false the caller must not propagate the
// effects any further down a replication sub-chain
bool Effects_Apply
(
	GraphContext *gc,          // graph to operate on
	const char *effects_buff,  // encoded effects
	size_t l                   // size of buffer
);

// create a new effects-buffer
EffectsBuffer *EffectsBuffer_New(void);

// reset effects-buffer
void EffectsBuffer_Reset
(
	EffectsBuffer *buff  // effects-buffer
);

// returns number of effects in buffer
uint64_t EffectsBuffer_Length
(
	const EffectsBuffer *buff  // effects-buffer
);

// get a copy of effectspbuffer internal buffer
unsigned char *EffectsBuffer_Buffer
(
	const EffectsBuffer *eb,  // effects-buffer
	size_t *n                 // size of returned buffer
);

// add a node creation effect to buffer
void EffectsBuffer_AddCreateNodeEffect
(
	EffectsBuffer *buff,    // effect buffer
	const Node *n,          // node created
	const LabelID *labels,  // node labels
	ushort label_count      // number of labels
);

// add a edge creation effect to buffer
void EffectsBuffer_AddCreateEdgeEffect
(
	EffectsBuffer *buff,  // effect buffer
	const Edge *edge      // edge created
);

// add a node deletion effect to buffer
void EffectsBuffer_AddDeleteNodeEffect
(
	EffectsBuffer *buff,  // effect buffer
	const Node *node      // node deleted
);

// add a edge deletion effect to buffer
void EffectsBuffer_AddDeleteEdgeEffect
(
	EffectsBuffer *buff,  // effect buffer
	const Edge *edge      // edge deleted
);

// add an entity attribute removal effect to buffer
void EffectsBuffer_AddEntityRemoveAttributeEffect
(
	EffectsBuffer *buff,         // effect buffer
	GraphEntity *entity,         // updated entity ID
	AttributeID attr_id,         // updated attribute ID
	GraphEntityType entity_type  // entity type
);

// add an entity add new attribute effect to buffer
void EffectsBuffer_AddEntityAddAttributeEffect
(
	EffectsBuffer *buff,         // effect buffer
	GraphEntity *entity,         // updated entity ID
	AttributeID attr_id,         // updated attribute ID
	SIValue value,               // value
	GraphEntityType entity_type  // entity type
);

// add an entity update attribute effect to buffer
void EffectsBuffer_AddEntityUpdateAttributeEffect
(
	EffectsBuffer *buff,         // effect buffer
	GraphEntity *entity,         // updated entity ID
	AttributeID attr_id,         // updated attribute ID
 	SIValue value,               // value
	GraphEntityType entity_type  // entity type
);

// records a SET_LABELS effect into the buffer:
// writes the effect type followed by the serialized node vector
void EffectsBuffer_AddLabelsEffect
(
	EffectsBuffer *buff,  // effect buffer to write into
	GrB_Vector nodes      // nodes that received the label
);

// records a REMOVE_LABELS effect into the buffer:
// writes the effect type followed by the serialized node vector
void EffectsBuffer_AddRemoveLabelsEffect
(
	EffectsBuffer *buff,  // effect buffer to write into
	GrB_Vector     nodes  // nodes that lost the label
);

// add a schema addition effect to buffer
void EffectsBuffer_AddNewSchemaEffect
(
	EffectsBuffer *buff,      // effect buffer
	const char *schema_name,  // id of the schema
	SchemaType t              // type of the schema
);

// add an attribute addition effect to buffer
void EffectsBuffer_AddNewAttributeEffect
(
	EffectsBuffer *buff,  // effect buffer
	const char *attr      // attribute name
);

// add an index field creation effect to buffer
// label/relationship-type and attribute are identified by both their
// internal id (authoritative, used to drive the operation) and their name
// (cross-checked at the receiving end to detect divergence)
void EffectsBuffer_AddCreateIndexEffect
(
	EffectsBuffer *buff,   // effect buffer
	SchemaType st,         // schema type (node/edge)
	int label_id,          // label/relationship-type id
	const char *label,     // label/relationship-type name
	AttributeID attr_id,   // attribute id
	const char *attr,      // attribute name
	IndexFieldType t,      // index field type (range/fulltext/vector)
	SIValue options        // index options
);

// add an index field deletion effect to buffer
void EffectsBuffer_AddDropIndexEffect
(
	EffectsBuffer *buff,   // effect buffer
	SchemaType st,         // schema type (node/edge)
	int label_id,          // label/relationship-type id
	const char *label,     // label/relationship-type name
	AttributeID attr_id,   // attribute id
	const char *attr,      // attribute name
	IndexFieldType t       // index field type (range/fulltext/vector)
);

// add a constraint creation effect to buffer
void EffectsBuffer_AddCreateConstraintEffect
(
	EffectsBuffer *buff,          // effect buffer
	ConstraintType ct,            // constraint type (unique/mandatory)
	GraphEntityType et,           // entity type (node/edge)
	int label_id,                 // label/relationship-type id
	const char *label,            // label/relationship-type name
	const AttributeID *attr_ids,  // constrained attribute ids
	const char **attrs,           // constrained attribute names
	uint8_t n                     // number of constrained attributes
);

// add a constraint deletion effect to buffer
void EffectsBuffer_AddDropConstraintEffect
(
	EffectsBuffer *buff,          // effect buffer
	ConstraintType ct,            // constraint type (unique/mandatory)
	GraphEntityType et,           // entity type (node/edge)
	int label_id,                 // label/relationship-type id
	const char *label,            // label/relationship-type name
	const AttributeID *attr_ids,  // constrained attribute ids
	const char **attrs,           // constrained attribute names
	uint8_t n                     // number of constrained attributes
);

// free effects-buffer
void EffectsBuffer_Free
(
	EffectsBuffer *eb
);

