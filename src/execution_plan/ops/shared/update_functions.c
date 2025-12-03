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

// commits delayed updates
void CommitUpdates
(
	GraphContext *gc,
	dict *updates,
	EntityType type
) {
	ASSERT(gc      != NULL);
	ASSERT(updates != NULL);
	ASSERT(type    != ENTITY_UNKNOWN);

	uint update_count         = HashTableElemCount(updates);
	bool constraint_violation = false;

	// return early if no updates are enqueued
	if(update_count == 0) return;

	dictEntry *entry;
	dictIterator *it = HashTableGetIterator(updates);
	MATRIX_POLICY policy = Graph_GetMatrixPolicy(gc->g);
	Graph_SetMatrixPolicy(gc->g, SYNC_POLICY_NOP);

	while((entry = HashTableNext(it)) != NULL) {
		PendingUpdateCtx *update = HashTableGetVal(entry);

		// if entity has been deleted, perform no updates
		if(GraphEntity_IsDeleted(update->ge)) {
			continue ;
		}

		AttributeSet old_set = GraphEntity_GetAttributes (update->ge) ;
		AttributeSet_TransferOwnership (old_set, update->attributes) ;

		// update the attributes on the graph entity
		GraphHub_UpdateEntityProperties(gc, update->ge, update->attributes,
				type == ENTITY_NODE ? GETYPE_NODE : GETYPE_EDGE, true);
		update->attributes = NULL;

		if(type == ENTITY_NODE) {
			GraphHub_UpdateNodeLabels(gc, (Node*)update->ge, update->add_labels,
				update->remove_labels, array_len(update->add_labels),
				array_len(update->remove_labels), true);
		}

		//----------------------------------------------------------------------
		// enforce constraints
		//----------------------------------------------------------------------

		if(constraint_violation == false) {
			// retrieve labels/rel-type
			uint label_count = 1;
			if (type == ENTITY_NODE) {
				label_count = Graph_LabelTypeCount(gc->g);
			}
			LabelID labels[label_count];
			if (type == ENTITY_NODE) {
				label_count = Graph_GetNodeLabels(gc->g, (Node*)update->ge, labels,
						label_count);
			} else {
				labels[0] = Edge_GetRelationID((Edge*)update->ge);
			}

			SchemaType stype = type == ENTITY_NODE ? SCHEMA_NODE : SCHEMA_EDGE;
			for(uint i = 0; i < label_count; i ++) {
				Schema *s = GraphContext_GetSchemaByID(gc, labels[i], stype);
				// TODO: a bit wasteful need to target relevant constraints only
				char *err_msg = NULL;
				if(!Schema_EnforceConstraints(s, update->ge, &err_msg)) {
					// constraint violation
					ASSERT(err_msg != NULL);
					constraint_violation = true;
					ErrorCtx_SetError("%s", err_msg);
					free(err_msg);
					break;
				}
			}
		}
	}
	Graph_SetMatrixPolicy(gc->g, policy);
	HashTableReleaseIterator(it);
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

// build pending updates in the 'updates' array to match all
// AST-level updates described in the context
// NULL values are allowed in SET clauses but not in MERGE clauses
void EvalEntityUpdates
(
	GraphContext *gc,
	dict *node_updates,
	dict *edge_updates,
	const Record r,
	const EntityUpdateEvalCtx *ctx,
	bool allow_null
) {
	//--------------------------------------------------------------------------
	// validate entity type
	//--------------------------------------------------------------------------

	// get the type of the entity to update
	// if the expected entity was not found, make no updates but do not error
	RecordEntryType t = Record_GetType (r, ctx->record_idx) ;
	if (unlikely (t == REC_TYPE_UNKNOWN)) {
		return ;
	}

	// make sure we're updating either a node or an edge
	if (unlikely (t != REC_TYPE_NODE && t != REC_TYPE_EDGE)) {
		ErrorCtx_RaiseRuntimeException (
			"Update error: alias '%s' did not resolve to a graph entity",
			ctx->alias) ;
	}

	// label(s) update can only be performed on nodes
	if (unlikely ((ctx->add_labels != NULL || ctx->remove_labels != NULL) &&
			t != REC_TYPE_NODE)) {
		ErrorCtx_RaiseRuntimeException (
				"Type mismatch: expected Node but was Relationship") ;
	}

	// get the updated entity
	GraphEntity *entity = Record_GetGraphEntity (r, ctx->record_idx) ;

	// if the entity is marked as deleted, make no updates but do not error
	if (unlikely (Graph_EntityIsDeleted (entity))) {
		return ;
	}

	dict *updates ;
	GraphEntityType entity_type ;

	if (t == REC_TYPE_NODE) {
		updates     = node_updates ;
		entity_type = GETYPE_NODE ;
	} else {
		updates     = edge_updates ;
		entity_type = GETYPE_EDGE ;
	}

	// do we already computed updates for this entity ?
	PendingUpdateCtx *update ;
	dictEntry *entry = HashTableFind (updates, (void *)ENTITY_GET_ID (entity));
	if (entry == NULL) {
		// create a new update context
		update = rm_malloc (sizeof (PendingUpdateCtx)) ;
		update->ge            = entity ;
		update->attributes    = AttributeSet_ShallowClone (*entity->attributes) ;
		update->add_labels    = NULL ;
		update->remove_labels = NULL ;
		// add update context to updates dictionary
		HashTableAdd (updates, (void *)ENTITY_GET_ID (entity), update) ;
	} else {
		// update context already exists
		update = (PendingUpdateCtx *)HashTableGetVal (entry) ;
	}

	if (array_len (ctx->add_labels) > 0) {
	   if (update->add_labels == NULL) {
		   update->add_labels =
			   array_new (const char *, array_len (ctx->add_labels)) ;
	   }
		array_union (update->add_labels, ctx->add_labels, strcmp) ;
	}

	if (array_len(ctx->remove_labels) > 0) {
		if (update->remove_labels == NULL) {
			update->remove_labels =
				array_new (const char *, array_len (ctx->remove_labels)) ;
		}
		array_union (update->remove_labels, ctx->remove_labels, strcmp) ;
	}

	AttributeSet *old_attrs = entity->attributes ;  // backup original attributes
	entity->attributes = &update->attributes ;      // assign shallow clone

	// now that the attribute-set was updated re-assign entity to the record
	if (t == REC_TYPE_NODE) {
		Record_AddNode (r, ctx->record_idx, *(Node *)entity) ;
	} else {
		Record_AddEdge (r, ctx->record_idx, *(Edge *)entity) ;
	}

	// if we're converting a SET clause, NULL is acceptable
	// as it indicates a deletion
	SIType accepted_properties = SI_VALID_PROPERTY_VALUE ;
	if (allow_null) {
		accepted_properties |= T_NULL ;
	}

	bool error = false;
	uint exp_count = array_len (ctx->properties) ;
	EffectsBuffer *eb = QueryCtx_GetEffectsBuffer () ;

	// evaluate each assigned expression
	// e.g. n.v = n.a + 2
	//
	// validate each new value type
	// e.g. invalid n.v = [1, {}]
	//
	// collect all updates into a single attribute-set

	// accumulated updates
	uint16_t n_updates = 0 ;            // number of updates
	SIValue attr_vals    [exp_count] ;  // attribute values
	AttributeID attr_ids [exp_count] ;  // attribute ids

	for (uint i = 0; i < exp_count && !error; i++) {
		PropertySetCtx *property = ctx->properties + i ;

		SIValue     v         = AR_EXP_Evaluate (property->exp, r) ;
		SIType      t         = SI_TYPE (v) ;
		UPDATE_MODE mode      = property->mode ;
		const char* attribute = property->attr_name ;

		//----------------------------------------------------------------------
		// n.v = 2
		//----------------------------------------------------------------------

		if (attribute != NULL) {

			// resolve attribute id
			if (property->attr_id == ATTRIBUTE_ID_NONE) {
				property->attr_id =
					GraphHub_FindOrAddAttribute (gc, attribute, true) ;
			}
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

		_FlushAccumulatedUpdates (eb, entity, entity_type, attr_vals, attr_ids,
				&n_updates) ;

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
		Record_AddNode (r, ctx->record_idx, *(Node *)entity) ;
	} else {
		Record_AddEdge (r, ctx->record_idx, *(Edge *)entity) ;
	}
}

void PendingUpdateCtx_Free
(
	PendingUpdateCtx *ctx
) {
	AttributeSet_Free(&ctx->attributes);
	array_free(ctx->add_labels);
	array_free(ctx->remove_labels);
	rm_free(ctx);
}

