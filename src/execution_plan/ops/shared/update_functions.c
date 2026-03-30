/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "update_functions.h"
#include "../../../query_ctx.h"
#include "../../../datatypes/map.h"
#include "../../../errors/errors.h"
#include "../../../datatypes/array.h"
#include "../../../graph/graph_hub.h"
#include "../../../graph/entities/node.h"

static bool _ValidateAttrType
(
	SIType accepted_properties,
	SIValue v
) {
	//--------------------------------------------------------------------------
	// validate value type
	//--------------------------------------------------------------------------

	SIType t = SI_TYPE(v);

	// make sure value is of an acceptable type
	if(!(t & accepted_properties)) {
		return false;
	}

	// in case of an array, make sure each element is of an
	// acceptable type
	if(t == T_ARRAY) {
		SIType invalid_properties = ~SI_VALID_PROPERTY_VALUE;
		return !SIArray_ContainsType(v, invalid_properties);
	}

	return true;
}

static void _WriteUpdatesToEffectsBuffer
(
	EffectsBuffer *eb,                // effects buffer
	GraphEntity *e,                   // updated entity
	GraphEntityType et,               // entity type
	AttributeID *attr_ids,            // attribute ids
	SIValue *attr_vals,               // attribute values
	AttributeSetChangeType *changes,  // changes
	uint16_t n                        // number of changes
) {
	// add changes to effects buffer
	for (uint i = 0; i < n; i++) {
		SIValue     v  = attr_vals[i] ;
		AttributeID id = attr_ids [i] ;

		switch (changes[i]) {
			case CT_DEL:
				// attribute removed
				EffectsBuffer_AddEntityRemoveAttributeEffect (eb, e, id, et) ;
				break;

			case CT_ADD:
				// attribute added
				EffectsBuffer_AddEntityAddAttributeEffect (eb, e, id, v, et) ;
				break;

			case CT_UPDATE:
				// attribute update
				EffectsBuffer_AddEntityUpdateAttributeEffect (eb, e, id, v, et) ;
				break;

			case CT_NONE:
				// no change
				break;

			default:
				assert ("unknown change type value" && false) ;
				break;
		}
	}
}

static void _ClearAttributeSet
(
	GraphEntity *e,
	GraphEntityType t,
	EffectsBuffer *eb
) {
	ASSERT (e  != NULL) ;
	ASSERT (eb != NULL) ;

	// free attribute-set and enqueue a 'clear' update
	EffectsBuffer_AddEntityRemoveAttributeEffect (eb, e, ATTRIBUTE_ID_ALL, t) ;
	AttributeSet_Free (e->attributes) ;
}

static void _AttributeSetUpdate
(
	GraphEntity *e,         // updated entity
	GraphEntityType et,     // entity type
	AttributeID *attr_ids,  // attribute ids
	SIValue *attr_vals,     // attribute values
	uint16_t n,             // number of attributes
	UPDATE_MODE mode,       // update mode replace / add
	EffectsBuffer *eb       // effects buffer
) {
	ASSERT (e         != NULL) ;
	ASSERT (eb        != NULL) ;
	ASSERT (attr_ids  != NULL) ;
	ASSERT (attr_vals != NULL) ;

	if (unlikely (n == 0)) {
		return ;
	}

	if (GraphEntity_GetAttributes (e) == NULL) {

		// add all attributes in one go
		AttributeSet_Add (e->attributes, attr_ids, attr_vals, n, true) ;

		// create effects
		for (uint i = 0; i < n; i++) {
			// attribute added effect
			EffectsBuffer_AddEntityAddAttributeEffect (eb, e, attr_ids[i],
					attr_vals[i], et) ;
		}

		// done
		return ;
	}

	//--------------------------------------------------------------------------
	// merge map attribute's into entity's attribute-set
	//--------------------------------------------------------------------------

	AttributeSetChangeType changes[n] ;
	AttributeSet_Update (changes, e->attributes, attr_ids, attr_vals, n, true) ;

	// write changes to effects buffer
	_WriteUpdatesToEffectsBuffer (eb, e, et, attr_ids, attr_vals, changes, n) ;
}

static void _FlushAccumulatedUpdates
(
	EffectsBuffer *eb,      // effects buffer
	GraphEntity *e,         // entity
	GraphEntityType et,     // entity type
	SIValue *attr_vals,     // attribute values
	AttributeID *attr_ids,  // attribute ids
	uint16_t *n             // number of updates
) {
	ASSERT (n         != NULL) ;
	ASSERT (e         != NULL) ;
	ASSERT (eb        != NULL) ;
	ASSERT (attr_ids  != NULL) ;
	ASSERT (attr_vals != NULL) ;

	if (*n == 0) {
		return ;
	}

	// update cloned attribute set, original remained untouched!
	AttributeSetChangeType changes[*n] ;
	AttributeSet_Update (changes, e->attributes, attr_ids, attr_vals, *n, true) ;

	// write changes to effects buffer
	_WriteUpdatesToEffectsBuffer (eb, e, et, attr_ids, attr_vals, changes, *n) ;

	// free values
	for (uint16_t i = 0; i < *n; i++) {
		SIValue_Free (attr_vals[i]) ;
	}

	*n = 0 ;
}

// update e's attribute-set from a map
static void _UpdateSetFromMap
(
	GraphContext *gc,       // graph context
	GraphEntity *e,         // entity
	GraphEntityType et,     // entity type
	UPDATE_MODE mode,       // update mode replace / merge
	EffectsBuffer *eb,      // effects buffer
	SIValue map,            // map to turn into attribute-set
	SIType accepted_types,  // accepted attribute types
	bool *error             // [output] error
) {
	ASSERT (e     != NULL) ;
	ASSERT (gc    != NULL) ;
	ASSERT (eb    != NULL) ;
	ASSERT (error != NULL) ;
	ASSERT (SI_TYPE (map) == T_MAP) ;

	// value is of type map e.g.
	// e =  {a:1, b:2}
	// e += {a:1, b:2}

	uint n = Map_KeyCount (map) ;
	if (unlikely (n == 0)) {
		return ;
	}

	uint16_t idx = 0;
	AttributeID attr_ids  [n] ;
	SIValue     attr_vals [n] ;

	//--------------------------------------------------------------------------
	// collect attributes from map
	//--------------------------------------------------------------------------

	// if entity has no attributes treat update as an add
	bool add = (GraphEntity_GetAttributes (e) == NULL) ;

	for (uint i = 0; i < n; i++) {
		SIValue key ;
		Map_GetIdx (map, i, &key, attr_vals + idx) ;

		// in "add" mode skip NULLs
		if (unlikely (add && SIValue_IsNull (attr_vals[idx]))) {
			continue ;
		}

		if (!_ValidateAttrType (accepted_types, attr_vals[idx])) {
			*error = true ;
			Error_InvalidPropertyValue() ;
			return ;
		}

		attr_ids[idx] = GraphHub_FindOrAddAttribute (gc, key.stringval, true) ;
		idx++ ;
	}

	_AttributeSetUpdate (e, et, attr_ids, attr_vals, idx , mode, eb) ;
}

// update e's attribute-set from e's
static void _UpdateSetFromEntity
(
	GraphEntity *s,      // entity to extract attributes from
	GraphEntity *e,      // entity to update
	GraphEntityType et,  // entity type
	UPDATE_MODE mode,    // update mode replace / merge
	EffectsBuffer *eb    // effects buffer
) {
	AttributeSet set = GraphEntity_GetAttributes (s) ;
	uint16_t n = AttributeSet_Count (set) ;
	if (unlikely (n == 0)) {
		return ;
	}

	if (mode == UPDATE_REPLACE) {
		// e's attribute-set should have been cleared
		ASSERT (*e->attributes == NULL) ;

		// set e's attribute-set with a clone of s
		*e->attributes = AttributeSet_Clone (set) ;
		set = GraphEntity_GetAttributes (e) ;

		// create effects
		for (uint i = 0; i < n; i++) {
			// attribute added effect
			SIValue v ;
			AttributeID attr ;
			AttributeSet_GetIdx (set, i, &attr, &v) ;
			EffectsBuffer_AddEntityAddAttributeEffect (eb, e, attr,
					v, et) ;
		}

		return ;
	}

	AttributeID attr_ids  [n] ;
	SIValue     attr_vals [n] ;

	// collect attributes
	for (uint i = 0; i < n; i++) {
		AttributeSet_GetIdx (set, i, attr_ids + i, attr_vals + i) ;
	}

	_AttributeSetUpdate (e, et, attr_ids, attr_vals, n , mode, eb) ;
}

// make sure label matrices used in SET n:L
// are of the correct dimensions NxN
void ensureMatrixDim
(
	GraphContext *gc,
	rax *blueprints
) {
	ASSERT (gc         != NULL) ;
	ASSERT (blueprints != NULL) ;

	bool resize_required = false ;
	Graph *g = GraphContext_GetGraph (gc) ;

	// set matrix sync policy to resize
	MATRIX_POLICY policy = Graph_SetMatrixPolicy (g, SYNC_POLICY_RESIZE) ;

	// for each update blueprint
	raxIterator it ;
	raxStart (&it, blueprints) ;
	raxSeek (&it, "^", NULL, 0) ;

	while (raxNext (&it)) {
		EntityUpdateDesc *ctx = it.data ;

		uint n = array_len (ctx->add_labels) ;
		for (uint i = 0 ; i < n ; i++) {
			const char *label = ctx->add_labels[i] ;
			const Schema *s = GraphContext_GetSchema (gc, label, SCHEMA_NODE) ;

			if (s != NULL) {
				// make sure label matrix is of the right dimensions
				resize_required = true ;
				Graph_GetLabelMatrix (g, Schema_GetID (s)) ;
			}
		}

		n = array_len (ctx->remove_labels) ;
		for (uint i = 0 ; i < n ; i++) {
			const char *label = ctx->remove_labels[i] ;
			const Schema *s = GraphContext_GetSchema (gc, label, SCHEMA_NODE) ;

			if (s != NULL) {
				// make sure label matrix is of the right dimensions
				resize_required = true ;
				Graph_GetLabelMatrix (g, Schema_GetID (s)) ;
			}
		}
	}

	raxStop (&it) ;

	// sync node labels matrix
	if (resize_required) {
		Graph_GetNodeLabelMatrix (g) ;
	}

	// restore matrix sync policy
	Graph_SetMatrixPolicy (g, policy) ;
}

void PendingUpdateCtx_Free
(
	PendingUpdateCtx *ctx
) {
	AttributeSet_Free (&ctx->attributes) ;
	rm_free (ctx) ;
}

