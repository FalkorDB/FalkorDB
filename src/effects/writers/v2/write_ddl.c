/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// the v2 DDL writers
//
// BYTE-FROZEN: shipped and read by engines this build will never see. The
// payload capture in .handover/capture_v2_payloads.py is what holds that.

#include "../../../RG.h"
#include "../../effects.h"
#include "../../effects_internal.h"
#include "write_v2.h"

void EffectsWriteV2_CreateIndex
(
	EffectsBuffer *buff,   // effect buffer
	SchemaType st,         // schema type (node/edge)
	int label_id,          // label/relationship-type id
	const char *label,     // label/relationship-type name
	AttributeID attr_id,   // attribute id
	const char *attr,      // attribute name
	IndexFieldType t,      // index field type (range/fulltext/vector)
	SIValue options,       // the v2 wire, written whole
	SIValue stated         // unused: v2 has no presence flags
) {
	//--------------------------------------------------------------------------
	// effect format:
	// effect type
	// schema type
	// label id
	// label name
	// attribute id
	// attribute name
	// index field type
	// options (map)
	//--------------------------------------------------------------------------

	EffectType eff_t = EFFECT_CREATE_INDEX ;

	EffectsBuffer_WriteBytes   (&eff_t, sizeof (eff_t), buff) ;
	EffectsBuffer_WriteBytes   (&st, sizeof (st), buff) ;
	EffectsBuffer_WriteBytes   (&label_id, sizeof (label_id), buff) ;
	EffectsBuffer_WriteString  (label, buff) ;
	EffectsBuffer_WriteBytes   (&attr_id, sizeof (attr_id), buff) ;
	EffectsBuffer_WriteString  (attr, buff) ;
	EffectsBuffer_WriteBytes   (&t, sizeof (t), buff) ;
	EffectsBuffer_WriteSIValue (&options, buff) ;

	EffectsBuffer_IncEffectCount (buff) ;
}

void EffectsWriteV2_DropIndex
(
	EffectsBuffer *buff,   // effect buffer
	SchemaType st,         // schema type (node/edge)
	int label_id,          // label/relationship-type id
	const char *label,     // label/relationship-type name
	AttributeID attr_id,   // attribute id
	const char *attr,      // attribute name
	IndexFieldType t       // index field type (range/fulltext/vector)
) {
	//--------------------------------------------------------------------------
	// effect format:
	// effect type
	// schema type
	// label id
	// label name
	// attribute id
	// attribute name
	// index field type
	//--------------------------------------------------------------------------

	EffectType eff_t = EFFECT_DROP_INDEX ;

	EffectsBuffer_WriteBytes  (&eff_t, sizeof (eff_t), buff) ;
	EffectsBuffer_WriteBytes  (&st, sizeof (st), buff) ;
	EffectsBuffer_WriteBytes  (&label_id, sizeof (label_id), buff) ;
	EffectsBuffer_WriteString (label, buff) ;
	EffectsBuffer_WriteBytes  (&attr_id, sizeof (attr_id), buff) ;
	EffectsBuffer_WriteString (attr, buff) ;
	EffectsBuffer_WriteBytes  (&t, sizeof (t), buff) ;

	EffectsBuffer_IncEffectCount (buff) ;
}

void EffectsWriteV2_CreateConstraint
(
	EffectsBuffer *buff,          // effect buffer
	ConstraintType ct,            // constraint type (unique/mandatory)
	GraphEntityType et,           // entity type (node/edge)
	uint32_t status,              // unused: v2 has no status field
	int label_id,                 // label/relationship-type id
	const char *label,            // label/relationship-type name
	const AttributeID *attr_ids,  // constrained attribute ids
	const char **attrs,           // constrained attribute names
	uint8_t n                     // number of constrained attributes
) {
	//--------------------------------------------------------------------------
	// effect format:
	// effect type
	// constraint type
	// entity type
	// label id
	// label name
	// attribute count
	// (attribute id, attribute name) pairs
	//--------------------------------------------------------------------------

	EffectType eff_t = EFFECT_CREATE_CONSTRAINT ;

	EffectsBuffer_WriteBytes (&eff_t, sizeof (eff_t), buff) ;

	EffectsBuffer_WriteBytes (&ct, sizeof (ct), buff) ;
	EffectsBuffer_WriteBytes (&et, sizeof (et), buff) ;
	EffectsBuffer_WriteBytes (&label_id, sizeof (label_id), buff) ;
	EffectsBuffer_WriteString (label, buff) ;

	EffectsBuffer_WriteBytes (&n, sizeof (n), buff) ;
	for (uint8_t i = 0; i < n; i++) {
		EffectsBuffer_WriteBytes (attr_ids + i, sizeof (AttributeID), buff) ;
		EffectsBuffer_WriteString (attrs [i], buff) ;
	}

	EffectsBuffer_IncEffectCount (buff) ;
}

void EffectsWriteV2_DropConstraint
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
	//--------------------------------------------------------------------------
	// effect format:
	// effect type
	// constraint type
	// entity type
	// label id
	// label name
	// attribute count
	// (attribute id, attribute name) pairs
	//--------------------------------------------------------------------------

	EffectType eff_t = EFFECT_DROP_CONSTRAINT ;

	EffectsBuffer_WriteBytes (&eff_t, sizeof (eff_t), buff) ;

	EffectsBuffer_WriteBytes (&ct, sizeof (ct), buff) ;
	EffectsBuffer_WriteBytes (&et, sizeof (et), buff) ;
	EffectsBuffer_WriteBytes (&label_id, sizeof (label_id), buff) ;
	EffectsBuffer_WriteString (label, buff) ;

	EffectsBuffer_WriteBytes (&n, sizeof (n), buff) ;
	for (uint8_t i = 0; i < n; i++) {
		EffectsBuffer_WriteBytes (attr_ids + i, sizeof (AttributeID), buff) ;
		EffectsBuffer_WriteString (attrs [i], buff) ;
	}

	EffectsBuffer_IncEffectCount (buff) ;
}
