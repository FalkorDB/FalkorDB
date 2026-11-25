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
	//
	// 'label_id' is an int rather than a LabelID because it is NEITHER type
	// on its own: 'st' decides whether it names a label or a relationship
	// type, and LabelID and RelationID are both int (node.h, edge.h). Naming
	// one of them here would be wrong for half the calls
	void (*CreateIndex)
	(
		EffectsBuffer *buff,    // effect buffer
		SchemaType st,          // SCHEMA_NODE or SCHEMA_EDGE
		int label_id,           // label id, or relationship-type id per 'st'
		const char *label,      // its name
		AttributeID attr_id,    // the indexed attribute
		const char *attr,       // its name
		IndexFieldType t,       // field type: range, fulltext, vector
		SIValue options,        // the field's full option map - v2 reads this
		SIValue stated          // only the options the statement named - v3
	);

	void (*DropIndex)
	(
		EffectsBuffer *buff,    // effect buffer
		SchemaType st,          // SCHEMA_NODE or SCHEMA_EDGE
		int label_id,           // label id, or relationship-type id per 'st'
		const char *label,      // its name
		AttributeID attr_id,    // the dropped attribute
		const char *attr,       // its name
		IndexFieldType t        // field type: range, fulltext, vector
	);

	// the CCH path index
	//
	// ONE EFFECT PER STATEMENT on both wires, not one per relationship type:
	// the index is defined over the types TOGETHER, and an effect naming a
	// subset of them describes a different index.
	void (*CreateCCH)
	(
		EffectsBuffer *buff,         // effect buffer
		const RelationID *rel_ids,   // relationship-type ids the index spans
		const char **rel_names,      // their names, parallel to 'rel_ids'
		uint rel_count,              // number of relationship types
		AttributeID weight_attr,     // edge weight attribute
		const char *weight_name      // its name
	);

	void (*DropCCH)
	(
		EffectsBuffer *buff,         // effect buffer
		const RelationID *rel_ids,   // relationship-type ids the index spans
		const char **rel_names,      // their names, parallel to 'rel_ids'
		uint rel_count,              // number of relationship types
		AttributeID weight_attr,     // edge weight attribute
		const char *weight_name      // its name
	);

	// constraint DDL
	//
	// 'status' is a ConstraintStatus that only v3 has a field for
	void (*CreateConstraint)
	(
		EffectsBuffer *buff,          // effect buffer
		ConstraintType ct,            // unique or mandatory
		GraphEntityType et,           // node or edge constraint
		uint32_t status,              // ConstraintStatus - v3 only
		int label_id,                 // label id, or relationship-type id
		const char *label,            // its name
		const AttributeID *attr_ids,  // the constrained attributes
		const char **attrs,           // their names, parallel to 'attr_ids'
		uint8_t n                     // number of attributes
	);

	void (*DropConstraint)
	(
		EffectsBuffer *buff,          // effect buffer
		ConstraintType ct,            // unique or mandatory
		GraphEntityType et,           // node or edge constraint
		int label_id,                 // label id, or relationship-type id
		const char *label,            // its name
		const AttributeID *attr_ids,  // the constrained attributes
		const char **attrs,           // their names, parallel to 'attr_ids'
		uint8_t n                     // number of attributes
	);
} EffectsWriter;

extern const EffectsWriter EFFECTS_WRITER_V2;
extern const EffectsWriter EFFECTS_WRITER_V3;
