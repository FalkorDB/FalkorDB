/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../effects_writer.h"

// the v3 write arms, declared here so writer_v3.c can assemble the table from
// arms defined across several files - the shape serializers/encoder/v19 uses

void EffectsWriteV3_CreateIndex
(
	EffectsBuffer *buff,
	SchemaType st,
	int label_id,
	const char *label,
	AttributeID attr_id,
	const char *attr,
	IndexFieldType t,
	SIValue options,
	SIValue stated
);

void EffectsWriteV3_DropIndex
(
	EffectsBuffer *buff,
	SchemaType st,
	int label_id,
	const char *label,
	AttributeID attr_id,
	const char *attr,
	IndexFieldType t
);

void EffectsWriteV3_CreateConstraint
(
	EffectsBuffer *buff,
	ConstraintType ct,
	GraphEntityType et,
	uint32_t status,
	int label_id,
	const char *label,
	const AttributeID *attr_ids,
	const char **attrs,
	uint8_t n
);

void EffectsWriteV3_DropConstraint
(
	EffectsBuffer *buff,
	ConstraintType ct,
	GraphEntityType et,
	int label_id,
	const char *label,
	const AttributeID *attr_ids,
	const char **attrs,
	uint8_t n
);

void EffectsWriteV3_CreateNode
(
	EffectsBuffer *buff,    // effect buffer
	const Node *n,          // node created
	const LabelID *labels,  // node labels
	ushort label_count      // number of labels
);

void EffectsWriteV3_CreateEdge
(
	EffectsBuffer *buff,  // effect buffer
	const Edge *edge      // edge created
);

void EffectsWriteV3_DeleteNode
(
	EffectsBuffer *buff,  // effect buffer
	const Node *node      // node deleted
);

void EffectsWriteV3_DeleteEdge
(
	EffectsBuffer *eb,  // effect buffer
	const Edge *edge    // edge deleted
);

void EffectsWriteV3_UpdateEntity
(
	EffectsBuffer *buff,         // effect buffer
	GraphEntity *entity,         // updated entity
	AttributeID attr_id,         // updated attribute, or ATTRIBUTE_ID_ALL
	SIValue value,               // value; a null is a removal
	GraphEntityType entity_type  // entity type
);

void EffectsWriteV3_Labels
(
	EffectsBuffer *buff,  // effect buffer
	GrB_Vector nodes,     // nodes the label applies to
	EffectType opcode     // SET_LABELS or REMOVE_LABELS
);

void EffectsWriteV3_NewSchema
(
	EffectsBuffer *buff,      // effect buffer
	const char *schema_name,  // name of the schema
	SchemaType st             // type of the schema
);

void EffectsWriteV3_NewAttribute
(
	EffectsBuffer *buff,  // effect buffer
	const char *attr      // attribute name
);
