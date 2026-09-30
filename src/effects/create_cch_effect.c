/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects.h"
#include "effects_internal.h"
#include "../util/arr.h"
#include "../schema/schema.h"
#include "../index/cch_index.h"
#include "../graph/graphcontext.h"

// add a CCH path-index creation effect to buffer
void EffectsBuffer_AddCreateCCHEffect
(
	EffectsBuffer    *buff,        // effect buffer
	const RelationID *rel_ids,     // relationship-type ids the index spans
	const char      **rel_names,   // relationship-type names (parallel to rel_ids)
	uint              rel_count,   // number of relationship types
	AttributeID       weight_attr, // weight attribute id
	const char       *weight_name  // weight attribute name
) {
	//--------------------------------------------------------------------------
	// effect format:
	//    effect type
	//    weight attribute id
	//    weight attribute name
	//    relationship-type count
	//    per relationship type: id, name
	//--------------------------------------------------------------------------

	EffectType eff_t = EFFECT_CREATE_CCH ;
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

// process CreateCCH effect: rebuild the definition-only CCH path index on this
// instance. relationship types + the weight attribute are cross-checked (id vs
// name) against local state before the (independently-derived) hierarchy is
// built. idempotent: an already-present identical index is left untouched.
//
// returns false if the replica has diverged from the master
bool ApplyCreateCCH
(
	FILE *stream,     // effects stream
	GraphContext *gc  // graph to operate on
) {
	AttributeID weight_attr ;
	fread_assert (&weight_attr, sizeof (weight_attr), stream) ;

	size_t l ;
	fread_assert (&l, sizeof (l), stream) ;
	char weight_name [l] ;
	fread_assert (weight_name, l, stream) ;

	uint rel_count ;
	fread_assert (&rel_count, sizeof (rel_count), stream) ;

	RelationID *rel_ids = arr_new (RelationID, rel_count) ;
	bool ok = true ;

	for (uint i = 0 ; i < rel_count ; i++) {
		RelationID rid ;
		fread_assert (&rid, sizeof (rid), stream) ;

		size_t rl ;
		fread_assert (&rl, sizeof (rl), stream) ;
		char rname [rl] ;
		fread_assert (rname, rl, stream) ;

		Schema *s = VerifySchema (gc, SCHEMA_EDGE, rid, rname) ;
		if (s == NULL) { ok = false ; break ; }
		arr_append (rel_ids, Schema_GetID (s)) ;
	}

	if (ok && !VerifyAttribute (gc, weight_attr, weight_name)) {
		ok = false ;
	}

	if (!ok) {
		arr_free (rel_ids) ;
		return false ;
	}

	// build the hierarchy locally unless an identical index already exists
	if (GraphContext_GetCCHIndex (gc, rel_ids, arr_len (rel_ids),
				weight_attr) == NULL) {
		CCHIndex *idx = CCHIndex_New (rel_ids, arr_len (rel_ids), weight_attr) ;
		CCHIndex_Build (idx, GraphContext_GetGraph (gc)) ;
		GraphContext_AddCCHIndex (gc, idx) ;
	}

	arr_free (rel_ids) ;
	return true ;
}
