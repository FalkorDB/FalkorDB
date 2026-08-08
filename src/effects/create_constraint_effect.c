/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects.h"
#include "effects_internal.h"
#include "../graph/graph_hub.h"

#include <stdio.h>

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

// process CreateConstraint effect
//
// returns true both when the constraint is genuinely created AND when it's
// a benign no-op (GraphHub_AddConstraint's CONSTRAINT_ALREADY_EXISTS status)
// - the latter is the expected shape of the async re-announcement issued
// once a pending constraint becomes active (see Constraint_Replicate),
// which every replica in the chain may legitimately receive more than once
//
// returns false only on genuine divergence from the master
bool ApplyCreateConstraint
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    constraint type
	//    entity type
	//    label id
	//    label name
	//    attribute count
	//    (attribute id, attribute name) pairs
	//--------------------------------------------------------------------------

	ConstraintType ct ;
	fread_assert (&ct, sizeof (ct), stream) ;

	GraphEntityType et ;
	fread_assert (&et, sizeof (et), stream) ;

	int label_id ;
	fread_assert (&label_id, sizeof (label_id), stream) ;

	size_t l ;
	fread_assert (&l, sizeof (l), stream) ;
	char label [l] ;
	fread_assert (label, l, stream) ;

	uint8_t n ;
	fread_assert (&n, sizeof (n), stream) ;

	AttributeID attr_ids [n] ;
	char *props [n] ;
	ReadConstraintAttributes (stream, n, attr_ids, props) ;

	//--------------------------------------------------------------------------
	// verify label & attributes against local state
	//--------------------------------------------------------------------------

	SchemaType st = (et == GETYPE_NODE) ? SCHEMA_NODE : SCHEMA_EDGE ;
	bool ok = (VerifySchema (gc, st, label_id, label) != NULL) ;

	for (uint8_t i = 0; ok && i < n; i++) {
		ok = VerifyAttribute (gc, attr_ids [i], props [i]) ;
	}

	bool result = false ;

	if (ok) {
		ConstraintCreateStatus status ;
		const char *err_msg = NULL ;
		Constraint c = GraphHub_AddConstraint (gc, ct, et, label,
				(const char **)props, n, false, &status, &err_msg) ;

		if (c != NULL) {
			Constraint_Enforce (c, (struct GraphContext *)gc) ;
			result = true ;
		} else if (status == CONSTRAINT_ALREADY_EXISTS) {
			result = true ;
		} else {
			RedisModule_Log (NULL, "warning",
					"GRAPH.EFFECT CREATE_CONSTRAINT failed: %s",
					err_msg != NULL ? err_msg : "unknown error") ;
		}
	}

	for (uint8_t i = 0; i < n; i++) {
		rm_free (props [i]) ;
	}

	return result ;
}

