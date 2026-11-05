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

// the CCH path index, create and drop
//
// Identical encodings behind two opcodes, which is why they share a writer.
//
// effect format:
//    effect type
//    weight attribute id
//    weight attribute name
//    relationship-type count
//    per relationship type: id, name
static void _v2_WriteCCH
(
	EffectsBuffer *buff,         // effect buffer
	EffectType eff_t,            // EFFECT_CREATE_CCH or EFFECT_DROP_CCH
	const RelationID *rel_ids,   // relationship-type ids the index spans
	const char **rel_names,      // their names, parallel to 'rel_ids'
	uint rel_count,              // number of relationship types
	AttributeID weight_attr,     // edge weight attribute
	const char *weight_name      // its name
) {
	EffectsBuffer_WriteBytes (&eff_t, sizeof (eff_t), buff) ;

	EffectsBuffer_WriteBytes (&weight_attr, sizeof (weight_attr), buff) ;
	EffectsBuffer_WriteString (weight_name, buff) ;

	EffectsBuffer_WriteBytes (&rel_count, sizeof (rel_count), buff) ;
	for (uint i = 0 ; i < rel_count ; i++) {
		EffectsBuffer_WriteBytes (&rel_ids [i], sizeof (rel_ids [i]), buff) ;
		EffectsBuffer_WriteString (rel_names [i], buff) ;
	}

	EffectsBuffer_IncEffectCount (buff) ;
}

void EffectsWriteV2_CreateCCH
(
	EffectsBuffer *buff,
	const RelationID *rel_ids,
	const char **rel_names,
	uint rel_count,
	AttributeID weight_attr,
	const char *weight_name
) {
	_v2_WriteCCH (buff, EFFECT_CREATE_CCH, rel_ids, rel_names, rel_count,
			weight_attr, weight_name) ;
}

void EffectsWriteV2_DropCCH
(
	EffectsBuffer *buff,
	const RelationID *rel_ids,
	const char **rel_names,
	uint rel_count,
	AttributeID weight_attr,
	const char *weight_name
) {
	_v2_WriteCCH (buff, EFFECT_DROP_CCH, rel_ids, rel_names, rel_count,
			weight_attr, weight_name) ;
}
