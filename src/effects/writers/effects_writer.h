/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../effects.h"

// the per-version write table, chosen once and never consulted again
//
// The split serializers/ uses on its read side: Decode_Previous switches on
// the encoding version once, and a version's decoder never asks again. Here
// the switch is EffectsBuffer_New, which already knows the version to emit.
//
// A table rather than a compile-time choice because EFFECTS_VERSION is a
// runtime config: a v3-capable build still has to emit v2 for an older
// replica.
typedef struct {
	// entity writers. The statistics each public entry point updates stay
	// there: they belong to both versions, so an arm must not own them
	void (*CreateNode)
	(
		EffectsBuffer *buff,    // effect buffer
		const Node *n,          // node created
		const LabelID *labels,  // node labels
		ushort label_count      // number of labels
	);

	void (*CreateEdge)
	(
		EffectsBuffer *buff,  // effect buffer
		const Edge *edge      // edge created
	);

	void (*DeleteNode)
	(
		EffectsBuffer *buff,  // effect buffer
		const Node *node      // node deleted
	);

	void (*DeleteEdge)
	(
		EffectsBuffer *eb,  // effect buffer
		const Edge *edge    // edge deleted
	);

	void (*UpdateEntity)
	(
		EffectsBuffer *buff,         // effect buffer
		GraphEntity *entity,         // updated entity
		AttributeID attr_id,         // updated attribute, or ATTRIBUTE_ID_ALL
		SIValue value,               // value; a null is a removal
		GraphEntityType entity_type  // entity type
	);

	void (*Labels)
	(
		EffectsBuffer *buff,  // effect buffer
		GrB_Vector nodes,     // nodes the label applies to
		EffectType opcode     // SET_LABELS or REMOVE_LABELS
	);

	void (*NewSchema)
	(
		EffectsBuffer *buff,      // effect buffer
		const char *schema_name,  // name of the schema
		SchemaType st             // type of the schema
	);

	void (*NewAttribute)
	(
		EffectsBuffer *buff,  // effect buffer
		const char *attr      // attribute name
	);

	// index DDL, one effect per FIELD on both wires
	//
	// 'options' is the v2 wire and 'stated' the subset the statement named;
	// each arm reads one of them and says which
	void (*CreateIndex)
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

	void (*DropIndex)
	(
		EffectsBuffer *buff,
		SchemaType st,
		int label_id,
		const char *label,
		AttributeID attr_id,
		const char *attr,
		IndexFieldType t
	);

	// constraint DDL
	//
	// 'status' is a ConstraintStatus that only v3 has a field for
	void (*CreateConstraint)
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

	void (*DropConstraint)
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
} EffectsWriter;

extern const EffectsWriter EFFECTS_WRITER_V2;
extern const EffectsWriter EFFECTS_WRITER_V3;
