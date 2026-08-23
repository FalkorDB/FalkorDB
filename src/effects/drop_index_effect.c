/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects.h"
#include "effects_internal.h"
#include "../graph/graph_hub.h"

#include <stdio.h>

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

// process DropIndex effect
// returns false if the replica has diverged from the master (the effect
// references label/attribute/index state that doesn't exist locally)
bool ApplyDropIndex
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    schema type
	//    label id
	//    label name
	//    attribute id
	//    attribute name
	//    index field type
	//--------------------------------------------------------------------------

	SchemaType st ;
	fread_assert (&st, sizeof (st), stream) ;

	int label_id ;
	fread_assert (&label_id, sizeof (label_id), stream) ;

	size_t l ;
	fread_assert (&l, sizeof (l), stream) ;
	char label [l] ;
	fread_assert (label, l, stream) ;

	AttributeID attr_id ;
	fread_assert (&attr_id, sizeof (attr_id), stream) ;

	fread_assert (&l, sizeof (l), stream) ;
	char attr [l] ;
	fread_assert (attr, l, stream) ;

	IndexFieldType t ;
	fread_assert (&t, sizeof (t), stream) ;

	//--------------------------------------------------------------------------
	// verify label & attribute against local state
	//--------------------------------------------------------------------------

	Schema *s = VerifySchema (gc, st, label_id, label) ;
	if (s == NULL) {
		return false ;
	}

	if (!VerifyAttribute (gc, attr_id, attr)) {
		return false ;
	}

	//--------------------------------------------------------------------------
	// confirm a matching index exists locally
	//--------------------------------------------------------------------------

	if (Schema_GetIndex (s, &attr_id, 1, t, true) == NULL) {
		RedisModule_Log (NULL, "warning",
				"GRAPH.EFFECT DROP_INDEX references index field '%s' on "
				"'%s' which doesn't exist locally", attr, label) ;
		return false ;
	}

	//--------------------------------------------------------------------------
	// drop index field
	//--------------------------------------------------------------------------

	return GraphHub_DropIndex (gc, st, label, attr, t, false) == INDEX_OK ;
}

