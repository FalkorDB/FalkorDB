/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../../graph/graphcontext.h"


static bool _EvalExpressions
(
	const PropertySetDesc *properties,  // expressions
	uint prop_count,       // expression count
	Record r,              // record
	SIValue *attr_vals,    // [input/output]
	AttributeID *attr_ids  // [input/output]
) {
	for (uint i = 0; i < prop_count ; i++) {
		PropertySetDesc *property = properties + i ;

		SIValue     v         = AR_EXP_Evaluate (property->exp, r) ;
		SIType      t         = SI_TYPE (v) ;
		UPDATE_MODE mode      = property->mode ;
		const char* attribute = property->attr_name ;

		//------------------------------------------------------------------
		// n.v = 2
		//------------------------------------------------------------------

		if (attribute != NULL) {
			ASSERT (property->attr_id != ATTRIBUTE_ID_NONE) :
				AttributeID attr_id = property->attr_id ;

			// accumulate update
			attr_vals [n_updates] = v;
			attr_ids  [n_updates] = attr_id;
			n_updates++ ;

			// validate attribute's type
			if (!_ValidateAttrType (accepted_properties, v)) {
				error = true ;
				// TODO: free accumulated updates
				Error_InvalidPropertyValue () ;
				break ;
			}
			continue ;
		}

		_FlushAccumulatedUpdates (eb, entity, entity_type, attr_vals,
				attr_ids, &n_updates) ;

		//----------------------------------------------------------------------
		// n = {v:2}, n = m
		//----------------------------------------------------------------------

		if (!(t & (T_NODE | T_EDGE | T_MAP))) {
			error = true ;
			SIValue_Free (v) ;
			Error_InvalidPropertyValue () ;
			break ;
		}

		//----------------------------------------------------------------------
		// n = {v:2}
		//----------------------------------------------------------------------

		if (t == T_MAP) {
			if (mode == UPDATE_REPLACE) {
				_ClearAttributeSet (entity, entity_type, eb) ;
			}

			_UpdateSetFromMap (gc, entity, entity_type, mode, eb, v,
					accepted_properties, &error) ;

			// free map
			SIValue_Free (v) ;
			continue ;
		}

		//----------------------------------------------------------------------
		// n = m
		//----------------------------------------------------------------------

		// value is a node or edge; perform attribute set reassignment
		ASSERT ((t & (T_NODE | T_EDGE))) ;

		GraphEntity *ge = v.ptrval ;

		// incase SET n = n / SET n += n
		if (unlikely (ENTITY_GET_ID (ge) == ENTITY_GET_ID (entity) &&
					((t == T_NODE && entity_type == GETYPE_NODE)   ||
					 (t == T_EDGE && entity_type == GETYPE_EDGE)))
		   ) {
			continue ;
		}

		if (mode == UPDATE_REPLACE) {
			_ClearAttributeSet (entity, entity_type, eb) ;
		}

		_UpdateSetFromEntity (ge, entity, entity_type, mode, eb) ;
	} // for loop end

}

// stage updates in the 'updates' context
// NULL values are allowed in SET clauses but not in MERGE clauses
void EvalUpdates
(
	GraphContext *gc,              // graph context
	StagedUpdatesCtx *ctx,         // staged updates context
	const Record *recs,            // records
	uint64_t n_recs,               // number of records
	const EntityUpdateDesc *desc,  // update descriptor
	bool allow_null                // allow nulls
) {
	uint exp_count = array_len (desc->properties) ;
	SIValue attr_vals    [exp_count] ;  // attribute values
	AttributeID attr_ids [exp_count] ;  // attribute ids

	bool update_labels =
		(desc->add_labels != NULL || desc->remove_labels != NULL) ;

	EffectsBuffer *eb = QueryCtx_GetEffectsBuffer () ;

	// if we're converting a SET clause, NULL is acceptable
	// as it indicates a deletion
	SIType accepted_properties = SI_VALID_PROPERTY_VALUE ;
	if (allow_null) {
		accepted_properties |= T_NULL ;
	}

	// make sure every updated attribute exists in the graph's schema
	for (uint i = 0 ; i < exp_count ; i++) {
		PropertySetDesc *property = desc->properties + i ;
		const char *attr_name = property->attr_name ;

		if (attr_name != NULL) {
			// resolve attribute id
			if (property->attr_id == ATTRIBUTE_ID_NONE) {
				property->attr_id =
					GraphHub_FindOrAddAttribute (gc, attr_name, true) ;
			}
		}
	}

	// foreach record:
	// 1. evaluate update expressions
	// 2. stage label addition / removal
	for (uint64_t i = 0 ; i < n_recs ; i++) {
		Record r = recs [i] ;

		//----------------------------------------------------------------------
		// validate entities type
		//----------------------------------------------------------------------

		// get the type of the entity to update
		// if the entity was not found, make no updates but do not error
		RecordEntryType t = Record_GetType (r, desc->record_idx) ;

		if (unlikely (t == REC_TYPE_UNKNOWN)) {
			continue ;
		}

		// make sure we're updating a graph entity
		if (unlikely (t != REC_TYPE_NODE && t != REC_TYPE_EDGE)) {
			ErrorCtx_RaiseRuntimeException (
				"Update error: alias '%s' did not resolve to a graph entity",
				desc->alias) ;
			return ;
		}

		// label(s) update can only be performed on nodes
		if (unlikely (update_labels && t != REC_TYPE_NODE)) {
			ErrorCtx_RaiseRuntimeException (
				"Label addition / removal can't be performed on an edge") ;
			return ;
		}

		// get the updated entity
		GraphEntity *entity = Record_GetGraphEntity (r, desc->record_idx) ;

		// if the entity is marked as deleted, make no updates but do not error
		if (unlikely (Graph_EntityIsDeleted (entity))) {
			// swap current record with the last one
			Record tmp = recs [n_recs - 1] ;
			recs [n_recs - 1] = r ;
			recs [i] = tmp ;
			n_recs -- ; 
			continue ;
		}

		dict *updates ;
		GraphEntityType entity_type ;

		if (t == REC_TYPE_NODE) {
			updates     = StagedUpdatesCtx_NodeUpdates (ctx) ;
			entity_type = GETYPE_NODE ;
		} else {
			updates     = StagedUpdatesCtx_EdgeUpdates (ctx) ;
			entity_type = GETYPE_EDGE ;
		}

		// did we already computed updates for this entity ?
		PendingUpdateCtx *update ;
		dictEntry *entry = HashTableFind (updates,
				(void *)ENTITY_GET_ID (entity)) ;

		if (entry == NULL) {
			// create a new update context
			update = rm_malloc (sizeof (PendingUpdateCtx)) ;
			update->ge         = entity ;
			update->attributes = AttributeSet_ShallowClone (*entity->attributes) ;
			// add update context to updates dictionary
			HashTableAdd (updates, (void *)ENTITY_GET_ID (entity), update) ;
		} else {
			// update context already exists
			update = (PendingUpdateCtx *) HashTableGetVal (entry) ;
		}

		AttributeSet *old_attrs = entity->attributes ;  // backup original attributes
		entity->attributes = &update->attributes ;      // assign shallow clone

		// re-assign entity to the record
		if (t == REC_TYPE_NODE) {
			Record_AddNode (r, desc->record_idx, *(Node *) entity) ;
		} else {
			Record_AddEdge (r, desc->record_idx, *(Edge *) entity) ;
		}

		bool error = false;

		// evaluate each expression
		// e.g. n.v = n.a + 2
		//
		// validate each value
		// e.g. invalid n.v = [1, {}]
		//
		// collect all updates into a single attribute-set

		// accumulated updates
		uint16_t n_updates = 0 ;  // number of updates

		for (uint j = 0; j < exp_count && !error; j++) {
			PropertySetDesc *property = desc->properties + j ;

			SIValue     v         = AR_EXP_Evaluate (property->exp, r) ;
			SIType      t         = SI_TYPE (v) ;
			UPDATE_MODE mode      = property->mode ;
			const char* attribute = property->attr_name ;

			//------------------------------------------------------------------
			// n.v = 2
			//------------------------------------------------------------------

			if (attribute != NULL) {
				ASSERT (property->attr_id != ATTRIBUTE_ID_NONE) :
				AttributeID attr_id = property->attr_id ;

				// accumulate update
				attr_vals [n_updates] = v;
				attr_ids  [n_updates] = attr_id;
				n_updates++ ;

				// validate attribute's type
				if (!_ValidateAttrType (accepted_properties, v)) {
					error = true ;
					// TODO: free accumulated updates
					Error_InvalidPropertyValue () ;
					break ;
				}
				continue ;
			}

			_FlushAccumulatedUpdates (eb, entity, entity_type, attr_vals,
					attr_ids, &n_updates) ;

			//----------------------------------------------------------------------
			// n = {v:2}, n = m
			//----------------------------------------------------------------------

			if (!(t & (T_NODE | T_EDGE | T_MAP))) {
				error = true ;
				SIValue_Free (v) ;
				Error_InvalidPropertyValue () ;
				break ;
			}

			//----------------------------------------------------------------------
			// n = {v:2}
			//----------------------------------------------------------------------

			if (t == T_MAP) {
				if (mode == UPDATE_REPLACE) {
					_ClearAttributeSet (entity, entity_type, eb) ;
				}

				_UpdateSetFromMap (gc, entity, entity_type, mode, eb, v,
						accepted_properties, &error) ;

				// free map
				SIValue_Free (v) ;
				continue ;
			}

			//----------------------------------------------------------------------
			// n = m
			//----------------------------------------------------------------------

			// value is a node or edge; perform attribute set reassignment
			ASSERT ((t & (T_NODE | T_EDGE))) ;

			GraphEntity *ge = v.ptrval ;

			// incase SET n = n / SET n += n
			if (unlikely (ENTITY_GET_ID (ge) == ENTITY_GET_ID (entity) &&
						((t == T_NODE && entity_type == GETYPE_NODE)   ||
						 (t == T_EDGE && entity_type == GETYPE_EDGE)))
			   ) {
				continue ;
			}

			if (mode == UPDATE_REPLACE) {
				_ClearAttributeSet (entity, entity_type, eb) ;
			}

			_UpdateSetFromEntity (ge, entity, entity_type, mode, eb) ;
		} // for loop end

		if (!error) {
			_FlushAccumulatedUpdates (eb, entity, entity_type, attr_vals, attr_ids,
					&n_updates) ;
		} else {
			// free accumulated updates
			for (uint16_t i = 0; i < n_updates; i++) {
				SIValue_Free (attr_vals[i]) ;
			}
		}

		// restore original attribute-set
		// changes should not be visible prior to the commit phase
		update->attributes = *entity->attributes ;
		entity->attributes = old_attrs ;
		if (t == REC_TYPE_NODE) {
			Record_AddNode (r, desc->record_idx, *(Node *)entity) ;
		} else {
			Record_AddEdge (r, desc->record_idx, *(Edge *)entity) ;
		}
	}
	if (array_len (desc->add_labels) > 0) {
		if (update->add_labels == NULL) {
			update->add_labels =
				array_new (const char *, array_len (desc->add_labels)) ;
		}
		array_union (update->add_labels, desc->add_labels, strcmp) ;
	}

	if (array_len(desc->remove_labels) > 0) {
		if (update->remove_labels == NULL) {
			update->remove_labels =
				array_new (const char *, array_len (desc->remove_labels)) ;
		}
		array_union (update->remove_labels, desc->remove_labels, strcmp) ;
	}
}

