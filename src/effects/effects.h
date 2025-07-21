/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "../graph/graphcontext.h"

#define EFFECTS_VERSION 1  // current effects encoding/decoding version

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
	EFFECT_ADD_SCHEMA,     // schema addition
	EFFECT_ADD_ATTRIBUTE,  // add attribute
} EffectType;

//------------------------------------------------------------------------------
// effects API
//------------------------------------------------------------------------------

// applys effects encoded in buffer
void Effects_Apply
(
	GraphContext *gc,          // graph to operate on
	const char *effects_buff,  // encoded effects
	size_t l                   // size of buffer
);

// create a new effects-buffer
EffectsBuffer *EffectsBuffer_New(void);

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
	Attribute_ID attr_id,        // updated attribute ID
	GraphEntityType entity_type  // entity type
);

// add an entity add new attribute effect to buffer
void EffectsBuffer_AddEntityAddAttributeEffect
(
	EffectsBuffer *buff,         // effect buffer
	GraphEntity *entity,         // updated entity ID
	Attribute_ID attr_id,        // updated attribute ID
	SIValue value,               // value
	GraphEntityType entity_type  // entity type
);

// add an entity update attribute effect to buffer
void EffectsBuffer_AddEntityUpdateAttributeEffect
(
	EffectsBuffer *buff,         // effect buffer
	GraphEntity *entity,         // updated entity ID
	Attribute_ID attr_id,        // updated attribute ID
 	SIValue value,               // value
	GraphEntityType entity_type  // entity type
);

// add a node add label effect to buffer
void EffectsBuffer_AddLabelsEffect
(
	EffectsBuffer *buff,     // effect buffer
	const Node *node,        // updated node
	const LabelID *lbl_ids,  // added labels
	size_t lbl_count         // number of removed labels
);

// add a node remove label effect to buffer
void EffectsBuffer_AddRemoveLabelsEffect
(
	EffectsBuffer *buff,     // effect buffer
	const Node *node,        // updated node
	const LabelID *lbl_ids,  // removed labels
	size_t lbl_count         // number of removed labels
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

// free effects-buffer
void EffectsBuffer_Free
(
	EffectsBuffer *eb
);

