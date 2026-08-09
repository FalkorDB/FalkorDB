/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects.h"
#include "effects_internal.h"
#include "../util/arr.h"
#include "../graph/graph_hub.h"
#include "../index/indexer.h"
#include "../commands/index_operations.h"

#include <stdio.h>

// add an index field creation effect to buffer
void EffectsBuffer_AddCreateIndexEffect
(
	EffectsBuffer *buff,   // effect buffer
	SchemaType st,         // schema type (node/edge)
	int label_id,          // label/relationship-type id
	const char *label,     // label/relationship-type name
	AttributeID attr_id,   // attribute id
	const char *attr,      // attribute name
	IndexFieldType t,      // index field type (range/fulltext/vector)
	SIValue options        // index options
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
	EffectsBuffer_WriteBytes (&eff_t, sizeof (eff_t), buff) ;

	EffectsBuffer_WriteBytes (&st, sizeof (st), buff) ;
	EffectsBuffer_WriteBytes (&label_id, sizeof (label_id), buff) ;
	EffectsBuffer_WriteString (label, buff) ;
	EffectsBuffer_WriteBytes (&attr_id, sizeof (attr_id), buff) ;
	EffectsBuffer_WriteString (attr, buff) ;
	EffectsBuffer_WriteBytes (&t, sizeof (t), buff) ;
	EffectsBuffer_WriteSIValue (&options, buff) ;

	EffectsBuffer_IncEffectCount (buff) ;
}

// process CreateIndex effect
//
// each CREATE INDEX statement emits one of these per field (see
// EffectsBuffer_AddCreateIndexEffect, emitted from GraphHub_AddIndex); a
// multi-field statement's schema/attribute effects interleave with its
// per-field CREATE_INDEX effects (a new attribute is only ever introduced
// once, right before the field that first references it), so - unlike
// ApplyCreateEdge/ApplyDeleteNode - effects of this type are NOT batched by
// stream adjacency. instead, index-level configuration (language/stopwords)
// and population are applied on every field independently, guarded to be
// idempotent: Index_SetLanguage tolerates being re-set to the same value,
// Index_SetStopwords is guarded by Index_ContainsStopwords since it isn't,
// and Index_Disable/Indexer_PopulateIndex's pending-changes counter is
// specifically designed to support being invoked once per field
//
// returns false if the replica has diverged from the master
bool ApplyCreateIndex
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
	//    options (map)
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

	SIValue options = SIValue_FromBinary (stream) ;

	//--------------------------------------------------------------------------
	// verify label & attribute against local state
	//--------------------------------------------------------------------------

	Schema *s = VerifySchema (gc, st, label_id, label) ;
	if (s == NULL) {
		SIValue_Free (options) ;
		return false ;
	}

	if (!VerifyAttribute (gc, attr_id, attr)) {
		SIValue_Free (options) ;
		return false ;
	}

	GraphEntityType et = (st == SCHEMA_NODE) ? GETYPE_NODE : GETYPE_EDGE ;

	//--------------------------------------------------------------------------
	// create index field
	//--------------------------------------------------------------------------

	Index idx = GraphHub_AddIndex (gc, label, attr, et, t, options, false) ;

	if (idx == NULL) {
		RedisModule_Log (NULL, "warning",
				"GRAPH.EFFECT CREATE_INDEX failed to create index field "
				"'%s' on '%s'", attr, label) ;
		SIValue_Free (options) ;
		return false ;
	}

	//--------------------------------------------------------------------------
	// apply index-level configuration (language / stopwords)
	//--------------------------------------------------------------------------

	char *language   = NULL ;
	char **stopwords = NULL ;
	IndexOperation_ExtractLevelConfig (&stopwords, &language, options) ;

	if (language != NULL) {
		Index_SetLanguage (idx, language) ;
	}

	if (stopwords != NULL) {
		if (!Index_ContainsStopwords (idx)) {
			Index_SetStopwords (idx, &stopwords) ;
		} else {
			// already configured by an earlier field of this same statement
			arr_free_cb (stopwords, rm_free) ;
		}
	}

	//--------------------------------------------------------------------------
	// populate index
	//--------------------------------------------------------------------------

	Index_Disable (idx) ;
	Indexer_PopulateIndex (gc, s, idx) ;

	SIValue_Free (options) ;

	return true ;
}

