/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// the v3 DDL writers
//
// None of these emit. v3 is one record per STATEMENT where C calls per FIELD,
// so a field is staged and the record is built when the query stops producing
// fields - see effects_v3_group.h.

#include "../../../RG.h"
#include "../../effects.h"
#include "../../effects_internal.h"
#include "write_v3.h"
#include "../../effects_v3_group.h"

void EffectsWriteV3_CreateIndex
(
	EffectsBuffer *buff,   // effect buffer
	SchemaType st,         // schema type (node/edge)
	int label_id,          // label/relationship-type id
	const char *label,     // label/relationship-type name
	AttributeID attr_id,   // attribute id
	const char *attr,      // attribute name
	IndexFieldType t,      // index field type (range/fulltext/vector)
	SIValue options,       // unused: pre-filled with defaults by now
	SIValue stated         // the subset the statement named
) {
	// 'stated', not 'options': v3's presence flag means "the statement said
	// this", and answering it from a map already carrying defaults would
	// announce options the user never wrote
	EffectsV3Grouping_AddIndexField (EffectsBuffer_V3 (buff),
			EFFECT_CREATE_INDEX, st, label_id, label, (uint32_t) t,
			attr_id, attr, stated) ;

	EffectsBuffer_IncEffectCount (buff) ;
}

void EffectsWriteV3_DropIndex
(
	EffectsBuffer *buff,   // effect buffer
	SchemaType st,         // schema type (node/edge)
	int label_id,          // label/relationship-type id
	const char *label,     // label/relationship-type name
	AttributeID attr_id,   // attribute id
	const char *attr,      // attribute name
	IndexFieldType t       // index field type (range/fulltext/vector)
) {
	// a drop carries no options at all - not an empty block
	EffectsV3Grouping_AddIndexField (EffectsBuffer_V3 (buff),
			EFFECT_DROP_INDEX, st, label_id, label, (uint32_t) t,
			attr_id, attr, SI_NullVal ()) ;

	EffectsBuffer_IncEffectCount (buff) ;
}

void EffectsWriteV3_CreateConstraint
(
	EffectsBuffer *buff,          // effect buffer
	ConstraintType ct,            // constraint type (unique/mandatory)
	GraphEntityType et,           // entity type (node/edge)
	uint32_t status,              // ConstraintStatus
	int label_id,                 // label/relationship-type id
	const char *label,            // label/relationship-type name
	const AttributeID *attr_ids,  // constrained attribute ids
	const char **attrs,           // constrained attribute names
	uint8_t n                     // number of constrained attributes
) {
	// the STATUS is what v2 has no field for: a replica never validates, so
	// the announcement is the only thing that can tell it an enforcing
	// constraint from one still building
	EffectsV3Grouping_AddConstraint (EffectsBuffer_V3 (buff),
			EFFECT_CREATE_CONSTRAINT, (uint32_t) ct, (uint32_t) et,
			status, label_id, label, attr_ids, attrs, n) ;

	EffectsBuffer_IncEffectCount (buff) ;
}

void EffectsWriteV3_DropConstraint
(
	EffectsBuffer *buff,          // effect buffer
	ConstraintType ct,            // constraint type (unique/mandatory)
	GraphEntityType et,           // entity type (node/edge)
	int label_id,                 // label/relationship-type id
	const char *label,            // label/relationship-type name
	const AttributeID *attr_ids,  // constrained attribute ids
	const char **attrs,           // constrained attribute names
	uint8_t n                     // number of constrained attributes
) {
	// a drop carries no status - there is nothing to converge on
	EffectsV3Grouping_AddConstraint (EffectsBuffer_V3 (buff),
			EFFECT_DROP_CONSTRAINT, (uint32_t) ct, (uint32_t) et,
			0, label_id, label, attr_ids, attrs, n) ;

	EffectsBuffer_IncEffectCount (buff) ;
}
