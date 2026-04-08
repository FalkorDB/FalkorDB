/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../../graph/graphcontext.h"

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
			Error_InvalidPropertyValue () ;
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


// evaluate updates for a single entity
//
static bool _UpdateEntity
(
	GraphEntity *e,             // entity being updated
	GraphEntityType et          // entity type
	StagedUpdatesCtx *staged,   // staged updates
	EntityUpdateDesc *desc,     // update descriptor
	Record r                    // record
) {
	ASSERT (e      != NULL) ;
	ASSERT (staged != NULL) ;
	ASSERT (desc   != NULL) ;
	ASSERT (et == GETYPE_NODE || et == GETYPE_EDGE) ;

	PendingUpdateCtx *ctx =
		StagedUpdatesCtx_GetEntityUpdateCtx (staged, e, et) ;
	ASSERT (ctx != NULL) ;

	AttributeSet *old_attrs = e->attributes ;  // backup original attributes
	e->attributes = &ctx->attributes ;         // assign shallow clone

	// re-assign entity to the record
	if (t == REC_TYPE_NODE) {
		Record_AddNode (r, desc->record_idx, *(Node *) e) ;
	} else {
		Record_AddEdge (r, desc->record_idx, *(Edge *) e) ;
	}

	PropertySetDesc *props = desc->properties ;
	ASSERT (props != NULL) ;

	uint32_t prop_count = array_length (props) ;
	ASSERT (prop_count > 0) ;

	uint32_t n_updates = 0 ;
	SIValue attr_vals     [prop_count] = {0} ;
	AttributeID  attr_ids [prop_count] = {0} ;

	for (uint32_t i = 0 ; i < prop_count ; i++) {
		PropertySetDesc *prop = props + i ;
		ASSERT (prop != NULL) ;

		SIValue     v         = AR_EXP_Evaluate (prop->exp, r) ;
		SIType      t         = SI_TYPE (v) ;
		UPDATE_MODE mode      = prop->mode ;
		const char* attribute = prop->attr_name ;

		//----------------------------------------------------------------------
		// n.v = 2
		//----------------------------------------------------------------------

		if (attribute != NULL) {
			AttributeID attr_id = prop->attr_id ;
			ASSERT (attr_id != ATTRIBUTE_ID_NONE) :

			// accumulate update
			attr_vals [n_updates] = v;
			attr_ids  [n_updates] = attr_id;
			n_updates++ ;

			// validate attribute's type
			if (!_ValidateAttrType (accepted_properties, v)) {
				error = true ;
				// TODO: free accumulated updates
				Error_InvalidPropertyValue () ;
				goto cleanup ;
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
			goto cleanup ;
		}

		//----------------------------------------------------------------------
		// n = {v:2}
		//----------------------------------------------------------------------

		if (t == T_MAP) {
			if (mode == UPDATE_REPLACE) {
				_ClearAttributeSet (e, et, eb) ;
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
	GraphContext *gc,                 // graph context
	StagedUpdatesCtx *staged_updates  // staged updates context
	const Record *recs,               // records
	uint32_t n_recs,                  // number of records
	EntityUpdateDesc **descs,         // update descriptors
	uint32_t n_descs,                 // number of update descriptors
	bool allow_null                   // allow nulls
) {
	if (n_recs == 0) {
		return ;
	}

	ASSERT (gc             != NULL) ;
	ASSERT (recs           != NULL) ;
	ASSERT (descs          != NULL) ;
	ASSERT (n_descs        > 0) ;
	ASSERT (staged_updates != NULL) ;

	// get the effects buffer
	// record every change in the effects buffer
	// used for replication and persistency
	EffectsBuffer *eb = QueryCtx_GetEffectsBuffer () ;

	//--------------------------------------------------------------------------
	// create missing labels & attribute IDs
	//--------------------------------------------------------------------------

	for (uint32_t i = 0 ; i < n_descs ; i++) {

		EntityUpdateDesc *desc = descs [i] ;
		uint32_t exp_count = array_len (desc->properties) ;

		//----------------------------------------------------------------------
		// create new attributes
		//----------------------------------------------------------------------

		for (uint32_t j = 0 ; j < exp_count ; j++) {
			PropertySetDesc *property = desc->properties + j ;
			const char *attr_name = property->attr_name ;

			if (attr_name != NULL && property->attr_id == ATTRIBUTE_ID_NONE) {
				// resolve attribute id
				property->attr_id =
					GraphHub_FindOrAddAttribute (gc, attr_name, true) ;
			}
		}

		//----------------------------------------------------------------------
		// create new labels
		//----------------------------------------------------------------------

		uint32_t add_lbl_count = array_length (desc->add_labels) ;
		for (uint32_t j = 0 ; j < add_lbl_count ; j++) {
			const char *lbl = desc->add_labels [j] ;
			ASSERT (lbl != NULL) ;

			if (GraphContext_GetSchema (gc, lbl, SCHEMA_NODE) == NULL) {
				GraphHub_AddSchema (gc, lbl,SCHEMA_NODE, true) ;
			}
		}

		//----------------------------------------------------------------------
		// remove non existing labels updates
		//----------------------------------------------------------------------

		// e.g.
		// REMOVE n:L
		// where the label `L` doesn't exists
		int32_t rmv_lbl_count = array_length (desc->remove_labels) ;
		for (int32_t j = 0 ; j < rmv_lbl_count ; j++) {
			const char *lbl = desc->remove_labels [j] ;
			ASSERT (lbl != NULL) ;

			if (GraphContext_GetSchema (gc, lbl, SCHEMA_NODE) == NULL) {
				array_del_fast (desc->remove_labels, j) ;
				j-- ;
				rmv_lbl_count-- ;
			}
		}
	}

	//--------------------------------------------------------------------------
	// 
	//--------------------------------------------------------------------------

	// if we're converting a SET clause, NULL is acceptable
	// as it indicates an attribute deletion
	SIType accepted_properties = SI_VALID_PROPERTY_VALUE ;
	if (allow_null) {
		accepted_properties |= T_NULL ;
	}

	for (uint32_t i = 0 ; i < n_descs ; i++) {

		EntityUpdateDesc *desc = descs [i] ;

		bool update_labels =
			(desc->add_labels != NULL || desc->remove_labels != NULL) ;

		// foreach record:
		// 1. evaluate update expressions
		// 2. stage label addition / removal
		for (uint32_t j = 0 ; j < n_recs ; j++) {
			Record r = recs [j] ;

			//------------------------------------------------------------------
			// validate entities type
			//------------------------------------------------------------------

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

			// if the entity is marked as deleted
			// make no updates but do not error
			if (unlikely (Graph_EntityIsDeleted (entity))) {
				continue ;
			}

			dict *updates ;
			GraphEntityType entity_type ;

			if (t == REC_TYPE_NODE) {
				updates     = StagedUpdatesCtx_NodeUpdates (staged_updates) ;
				entity_type = GETYPE_NODE ;
			} else {
				updates     = StagedUpdatesCtx_EdgeUpdates (staged_updates) ;
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

