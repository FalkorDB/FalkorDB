/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "update_functions.h"
#include "../../../query_ctx.h"
#include "../../../datatypes/map.h"
#include "../../../errors/errors.h"
#include "../../../datatypes/array.h"
#include "../../../graph/graph_hub.h"
#include "../../../graph/graphcontext.h"
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

<<<<<<< batch-label-update
=======
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
	bool enforce_constraints  = GraphContext_HasConstraints (gc) ;
	bool constraint_violation = false;

	// return early if no updates are enqueued
	if (update_count == 0) {
		return ;
	}

	dictEntry *entry ;
	dictIterator *it = HashTableGetIterator (updates) ;

	Graph *g = GraphContext_GetGraph (gc) ;
	MATRIX_POLICY policy = Graph_GetMatrixPolicy (g) ;
	Graph_SetMatrixPolicy (g, SYNC_POLICY_NOP) ;

	while ((entry = HashTableNext(it)) != NULL) {
		PendingUpdateCtx *update = HashTableGetVal (entry) ;

		// if entity has been deleted, perform no updates
		if (GraphEntity_IsDeleted (update->ge)) {
			continue ;
		}

		AttributeSet old_set = GraphEntity_GetAttributes (update->ge) ;
		AttributeSet_TransferOwnership (old_set, update->attributes) ;

		// update the attributes on the graph entity
		GraphHub_UpdateEntityProperties (gc, update->ge, update->attributes,
				type == ENTITY_NODE ? GETYPE_NODE : GETYPE_EDGE, true) ;
		update->attributes = NULL ;

		if (type == ENTITY_NODE) {
			GraphHub_UpdateNodeLabels (gc, (Node*)update->ge,
					update->add_labels, update->remove_labels,
					arr_len (update->add_labels),
					arr_len (update->remove_labels), true) ;
		}

		//----------------------------------------------------------------------
		// enforce constraints
		//----------------------------------------------------------------------

		if (enforce_constraints && constraint_violation == false) {
			// retrieve labels/rel-type
			uint label_count = 1 ;
			if (type == ENTITY_NODE) {
				label_count = Graph_LabelTypeCount (g) ;
			}
			LabelID labels[label_count] ;
			if (type == ENTITY_NODE) {
				label_count = Graph_GetNodeLabels (g, (Node*)update->ge,
						labels, label_count) ;
			} else {
				labels[0] = Edge_GetRelationID ((Edge*)update->ge) ;
			}

			SchemaType stype = type == ENTITY_NODE ? SCHEMA_NODE : SCHEMA_EDGE ;
			for (uint i = 0; i < label_count; i ++) {
				Schema *s = GraphContext_GetSchemaByID (gc, labels[i], stype) ;
				// TODO: a bit wasteful need to target relevant constraints only
				char *err_msg = NULL ;
				if (!Schema_EnforceConstraints (s, update->ge, &err_msg)) {
					// constraint violation
					ASSERT (err_msg != NULL) ;
					constraint_violation = true ;
					ErrorCtx_SetError ("%s", err_msg) ;
					free (err_msg) ;
					break ;
				}
			}
		}
	}

	Graph_SetMatrixPolicy (g, policy) ;
	HashTableReleaseIterator (it) ;
}

>>>>>>> master
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
            EffectsBuffer_AddEntityAddAttributeEffect (eb, e, attr_ids [i],
					attr_vals [i], et) ;
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

// applies a map of attributes to a graph entity's attribute-set
//
// iterates over every key-value pair in 'map', validates each value's type
// against 'accepted_types', resolves (or creates) the corresponding attribute
// ID, and forwards the collected pairs to _AttributeSetUpdate
//
// NULL values are skipped when the entity has no existing attribute-set
// because there is nothing to delete
// in all other cases NULL is forwarded
// as-is (meaning: delete that attribute), and it is the caller's
// responsibility to ensure 'accepted_types' permits T_NULL when that
// behaviour is intended
//
// returns:
//   true  - all attributes were processed and the entity was updated
//   false - a value in the map failed type validation; an error has been
//           raised in the error context and no partial update is applied
//
// note: the caller is responsible for clearing existing attributes before
// calling this function if full-replace (SET n = {…}) semantics are needed
static bool _UpdateSetFromMap
(
	GraphContext *gc,             // graph context
	GraphEntity *entity,          // entity whose attributes will be updated
	GraphEntityType entity_type,  // NODE or EDGE
	EffectsBuffer *eb,            // effects buffer; every attribute write is
								  // recorded here for replication and persistence

	SIValue map,                  // T_MAP value containing the new attribute
								  // key-value pairs

	SIType accepted_types         // accepted attribute types
) {
	// validations
	ASSERT (entity != NULL) ;
	ASSERT (gc     != NULL) ;
	ASSERT (eb     != NULL) ;
	ASSERT (SI_TYPE (map) == T_MAP) ;

	// handling update expression of type:
	// e =  {a:1, b:2}
	// or
	// e += {a:1, b:2}

	// quick return on empty map
	const uint16_t key_count = Map_KeyCount (map) ;
	if (unlikely (key_count == 0)) {
		return true ;
	}

	uint16_t attr_count = 0;
	AttributeID attr_ids  [key_count] ;
	SIValue     attr_vals [key_count] ;

	//--------------------------------------------------------------------------
	// collect attributes from map
	//--------------------------------------------------------------------------

	// if entity has no attributes treat update as an addition
	bool is_empty_entity = (GraphEntity_GetAttributes (entity) == NULL) ;
	const bool log = true ;

	for (uint16_t i = 0; i < key_count; i++) {
		SIValue key ;
		Map_GetIdx (map, i, &key, attr_vals + attr_count) ;

		// skip NULL values when entity doesn't have any attributes
		if (unlikely (is_empty_entity &&
					  SIValue_IsNull (attr_vals [attr_count]))) {
			continue ;
		}

		if (!_ValidateAttrType (accepted_types, attr_vals [attr_count])) {
			Error_InvalidPropertyValue () ;
			return false ;
		}

		// convert key to attribute-id, missing attributes will be created
		attr_ids [attr_count] =
			GraphHub_FindOrAddAttribute (gc, key.stringval, log) ;
		attr_count++ ;
	}

	// add attributes to entity
	if (attr_count > 0) {
		_AttributeSetUpdate (entity, entity_type, attr_ids, attr_vals,
				attr_count, eb) ;
	}

	return true ;
}

// copies attributes from one graph entity to another
//
// implements the attribute-transfer half of:
//   SET dest = src    (UPDATE_REPLACE)
//   SET dest += src   (UPDATE_MERGE)
//
// two distinct paths are taken depending on whether 'dest_entity' already
// has an attribute-set:
//
//   empty dest  — the source attribute-set is cloned directly onto the
//                 destination. Each attribute is then recorded in the effects
//                 buffer as an individual ADD effect
//
//   non-empty dest — attributes are collected from 'src_entity' and
//                    forwarded to _AttributeSetUpdate, which handles
//                    per-attribute merging, overwriting, and effect
//                    recording
//
// if 'src_entity' carries no attributes the function returns immediately
// with no modifications to 'dest_entity'
//
// Parameters:
//
// Returns: void
//
// Note: this function does not clear 'dest_entity's existing attributes
// before copying when mode == UPDATE_REPLACE; callers that need full-replace
// semantics (SET n = m) must clear the destination attribute-set beforehand.
//
// Note: like _UpdateSetFromMap, this function uses stack-allocated VLAs
// sized by 'attr_count'. Very large attribute-sets risk stack overflow.
static void _UpdateSetFromEntity
(
	const GraphEntity *src_entity,  // entity whose attributes are the source
									// of the update; not modified

	GraphEntity *dest_entity,       // entity whose attribute-set will be
									// updated in place

	GraphEntityType entity_type,    // NODE or EDGE
	EffectsBuffer *eb               // every attribute change is recorded here
									// for replication and persistence
) {
	AttributeSet set = GraphEntity_GetAttributes (src_entity) ;
	uint16_t attr_count = AttributeSet_Count (set) ;

	// early return if src entity doesn't have attributes
	if (unlikely (attr_count == 0)) {
		return ;
	}

	if (*dest_entity->attributes == NULL) {
		// set dest entity attribute-set to a clone of src entity attributes
		*dest_entity->attributes = AttributeSet_Clone (set) ;

		// re-fetch from dest so the effects loop holds pointers into the
		// cloned set, not the src's set which we no longer own
		set = GraphEntity_GetAttributes (dest_entity) ;

		// create effects
		for (uint16_t i = 0; i < attr_count; i++) {
			// attribute added effect
			SIValue val ;
			AttributeID attr_id ;
			AttributeSet_GetIdx (set, i, &attr_id, &val) ;
			EffectsBuffer_AddEntityAddAttributeEffect (eb, dest_entity, attr_id,
					val, entity_type) ;
		}

		return ;
	}

	AttributeID attr_ids  [attr_count] ;
	SIValue     attr_vals [attr_count] ;

	// collect attributes
	for (uint16_t i = 0; i < attr_count; i++) {
		AttributeSet_GetIdx (set, i, attr_ids + i, attr_vals + i) ;
	}

	_AttributeSetUpdate (dest_entity, entity_type, attr_ids, attr_vals,
			attr_count, eb) ;
}

// evaluates all update expressions in 'desc' and stages the resulting
// attribute changes on the pending attribute-set for 'entity'
//
//   MATCH (n)
//   UNWIND range(0, 2) AS x
//   SET n.v = n.v + 1       ← each iteration sees the previous iteration's write
//
// three categories of update expression are handled:
//
//   n.v = expr    — scalar attribute update; expressions are accumulated
//                   and flushed in a single _AttributeSetUpdate call for
//                   efficiency
//
//   n = {…}  /    — full-entity replace or merge from a map; accumulated
//   n += {…}        scalar updates are flushed first, then _UpdateSetFromMap
//                   is called. for UPDATE_REPLACE the attribute-set is
//                   cleared before the map is applied
//
//   n = m  /      — full-entity replace or merge from another graph entity;
//   n += m          same flush-then-apply pattern as the map case
//                   self-assignment (SET n = n) is detected and skipped
//
// the function guarantees that no partial update is visible if it returns
// false: accumulated SIValues are freed in the cleanup label
//
// returns:
//   true  - all expressions evaluated and staged successfully
//   false - a type validation error was raised; the error context has been
//           populated and no attribute changes have been applied
static bool _UpdateEntity
(
	GraphContext *gc,             // graph context
	GraphEntity *entity,          // entity whose attribute-set will be updated
	GraphEntityType entity_type,  // NODE or EDGE
	StagedUpdatesCtx *staged,     // staged updates context; holds the pending
								  // attribute-set for 'entity' so writes do not
								  // touch the live graph until the txn commits

	EntityUpdateDesc *desc,       // update descriptor produced by the planner;
								  // contains the list of PropertySetDesc
								  // entries to evaluate in order

	Record rec,                   // the current record; updated in place so
								  // that subsequent expressions in the same SET
								  // clause see the pending state

	SIType accepted_types,        // accepted attribute types

	EffectsBuffer *eb             // effects buffer; every attribute write is
								  // recorded here for replication and persistence
) {
	// validations
	ASSERT (rec    != NULL) ;
	ASSERT (desc   != NULL) ;
	ASSERT (entity != NULL) ;
	ASSERT (staged != NULL) ;
	ASSERT (entity_type == GETYPE_NODE || entity_type == GETYPE_EDGE) ;

	PropertySetDesc *props = desc->properties ;
	uint16_t prop_count = array_len (props) ;

	if (unlikely (prop_count == 0)) {
		// quick return, no updates to entity's attributes
		// e.g. MATCH (n) SET n:L
		return true ;
	}

	bool res = true ;

	PendingUpdateCtx *ctx =
		StagedUpdatesCtx_GetEntityUpdateCtx (staged, entity, entity_type) ;
	ASSERT (ctx != NULL) ;

	//------------------------------------------------------------------
	// backup original attributes
	// assign pending attributes
	//------------------------------------------------------------------

	AttributeSet *original_attrs = entity->attributes ;
	entity->attributes = &ctx->attributes ;

	// make pending attribute-set visible to record
	// such that update expressions will be able to see intermediate changes
	// e.g.
	//
	// MATCH (n)
	// UNWIND range (0, 2) AS x
	// SET n.v = n.v + 1

	if (entity_type == GETYPE_NODE) {
		Record_AddNode (rec, desc->record_idx, *(Node *) entity) ;
	} else {
		Record_AddEdge (rec, desc->record_idx, *(Edge *) entity) ;
	}

	uint16_t n_updates  = 0 ;
	AttributeID  attr_ids  [prop_count] ;
	SIValue      attr_vals [prop_count] ;

	memset (attr_ids,  0, sizeof (AttributeID) * prop_count) ;
	memset (attr_vals, 0, sizeof (SIValue)     * prop_count) ;

	// evaluate each update expression
	// e.g.
	// SET n.v = 2,     # attribute update
	//     n += $info,  # adding multiple attributes
	//     n = m        # replacing n's attributes with m's
	for (uint16_t i = 0 ; i < prop_count ; i++) {
		PropertySetDesc *prop = props + i ;
		ASSERT (prop != NULL) ;

		SIValue     v         = AR_EXP_Evaluate (prop->exp, rec) ;
		SIType      t         = SI_TYPE (v) ;
		UPDATE_MODE mode      = prop->mode ;
		const char *attribute = prop->attr_name ;

		// a replace operation e.g. `n = m` must be the first update expression
		ASSERT ((mode == UPDATE_REPLACE && i == 0) || mode != UPDATE_REPLACE) ;

		//----------------------------------------------------------------------
		// n.v = 2
		//----------------------------------------------------------------------

		if (attribute != NULL) {
			AttributeID attr_id = prop->attr_id ;
			ASSERT (attr_id != ATTRIBUTE_ID_NONE) ;

			// accumulate update
			attr_vals [n_updates] = v ;
			attr_ids  [n_updates] = attr_id ;
			n_updates ++ ;

			// validate attribute's type
			if (!_ValidateAttrType (accepted_types, v)) {
				// TODO: free accumulated updates
				Error_InvalidPropertyValue () ;
				res = false ;
				goto cleanup ;
			}

			continue ;
		}

		// flush accumulated attributes, preparing a clean slate for update
		// expression of type n += {v:2} and n += m
		if (n_updates > 0) {
			_FlushAccumulatedUpdates (eb, entity, entity_type, attr_vals,
					attr_ids, &n_updates) ;
		}

		//----------------------------------------------------------------------
		// n = {v:2}, n = m
		//----------------------------------------------------------------------

		if (!(t & (T_NODE | T_EDGE | T_MAP))) {
			SIValue_Free (v) ;
			Error_InvalidPropertyValue () ;
			res = false ;
			goto cleanup ;
		}

		//----------------------------------------------------------------------
		//  n = {v:2} or n += {v:2}
		//----------------------------------------------------------------------

		if (t == T_MAP) {
			if (mode == UPDATE_REPLACE) {
				_ClearAttributeSet (entity, entity_type, eb) ;
			}

			res = _UpdateSetFromMap (gc, entity, entity_type, eb, v,
					accepted_types) ;

			SIValue_Free (v) ;

			if (!res) {
				goto cleanup ;
			}

			continue ;
		}

		//----------------------------------------------------------------------
		// n = m or n += m
		//----------------------------------------------------------------------

		// value is a node or edge; perform attribute set reassignment
		ASSERT ((t & (T_NODE | T_EDGE))) ;

		GraphEntity *ge = v.ptrval ;

		// incase SET n = n or SET n += n
		if (unlikely (ENTITY_GET_ID (ge) == ENTITY_GET_ID (entity) &&
					((t == T_NODE && entity_type == GETYPE_NODE)   ||
					 (t == T_EDGE && entity_type == GETYPE_EDGE)))
		   ) {
			SIValue_Free (v) ;
			continue ;
		}

		if (mode == UPDATE_REPLACE) {
			_ClearAttributeSet (entity, entity_type, eb) ;
		}

		_UpdateSetFromEntity (ge, entity, entity_type, eb) ;
		SIValue_Free (v) ;

	} // for loop end

	// flush any remaining updates
	if (n_updates > 0) {
		_FlushAccumulatedUpdates (eb, entity, entity_type, attr_vals, attr_ids,
				&n_updates) ;
	}

cleanup:
	for (uint16_t i = 0 ; i < n_updates ; i++) {
		SIValue_Free (attr_vals [i]) ;
	}

	//------------------------------------------------------------------
	// restore original attribute-set
	// changes should not be visible prior to the commit phase
	//------------------------------------------------------------------

	ctx->attributes = *entity->attributes ;
	entity->attributes = original_attrs ;

	if (entity_type == GETYPE_NODE) {
		Record_AddNode (rec, desc->record_idx, *(Node *)entity) ;
	} else {
		Record_AddEdge (rec, desc->record_idx, *(Edge *)entity) ;
	}

	return res ;
}

// make sure every label and attribute name mentioned in the update descriptors
// exists within the graph context
// in addition non existing labels marked for removal e.g. REMOVE n:L
// are discarded as these will result in a no op
static void _UpdateSchemas
(
    GraphContext     *gc,              // graph context; used for schema lookups
						               // and attribute ID resolution
    EntityUpdateDesc **descs,          // array of update descriptors;
									   // one per updated alias, each carrying
									   // the ordered list of update expressions
    uint32_t n_descs                   // number of descriptors in 'descs'
) {
	const bool log = true ;

    // TODO: chnages to graph's schema shouldn't be visible to other threads
    // as we're not holding the write lock at this stage
	for (uint32_t i = 0 ; i < n_descs ; i++) {

	    EntityUpdateDesc *desc = descs [i] ;

		//----------------------------------------------------------------------
		// introduce missing attributes to graph's schema
		//----------------------------------------------------------------------

		uint32_t prop_count = array_len (desc->properties) ;
		for (uint32_t j = 0 ; j < prop_count ; j++) {
			PropertySetDesc *property = desc->properties + j ;
			const char *attr_name = property->attr_name ;

			if (attr_name != NULL && property->attr_id == ATTRIBUTE_ID_NONE) {
				// resolve attribute id
				property->attr_id =
					GraphHub_FindOrAddAttribute (gc, attr_name, log) ;
			}
		}

		//----------------------------------------------------------------------
		// introduce missing labels to graph's schema
		//----------------------------------------------------------------------

		uint32_t add_lbl_count = array_len (desc->add_labels) ;
		for (uint32_t j = 0 ; j < add_lbl_count ; j++) {
			const char *lbl = desc->add_labels [j] ;
			ASSERT (lbl != NULL) ;

			if (GraphContext_GetSchema (gc, lbl, SCHEMA_NODE) == NULL) {
				GraphHub_AddSchema (gc, lbl, SCHEMA_NODE, log) ;
			}
		}

		//----------------------------------------------------------------------
		// prune remove-label ops for labels not present in the schema
		//----------------------------------------------------------------------

		// e.g.
		// MATCH (n)
		// REMOVE n:L
		//
		// if the label doesn't exist in the schema, removing it is a no-op
		// drop the entry so downstream logic doesn't try to resolve
		// a nonexistent schema ID

		// signed: decremented during loop
		int32_t rmv_lbl_count = array_len (desc->remove_labels) ;
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
}

// evaluates and stages all pending graph updates described by 'descs' across
// every record in 'recs'
//
// the function operates in three sequential phases:
//
//   phase 1 — schema preparation
//     iterates over all descriptors and ensures that every attribute name and
//     label referenced by an update expression already exists in the graph
//     schema, creating missing ones as needed
//     label-removal ops that reference nonexistent labels are pruned from the
//     descriptor so downstream code never attempts to resolve unknown schema ID
//
//   phase 2 — expression evaluation and attribute staging
//     for each (descriptor, record) pair:
//       - validates that the target alias resolves to a live graph entity
//       - calls _UpdateEntity to evaluate every update expression in the
//         descriptor and write results into the pending attribute-set
//       - collects entity IDs for nodes that require label changes
//
//   phase 3 — Label staging
//     after all records for a descriptor have been processed, batches the
//     collected node IDs into StagedUpdatesCtx_LabelNodes /
//     StagedUpdatesCtx_UnLabelNodes
//
// NULL handling:
//   when 'allow_null' is true (SET clause) a NULL value signals attribute
//   deletion and is forwarded as-is to _UpdateEntity
//   when false (MERGE clause) NULL values are rejected as invalid
//
// error handling:
//   on any runtime error the function jumps to 'cleanup', frees the
//   node_ids array, and returns
//   the error has already been raised by the time cleanup runs
bool EvalUpdates
(
    GraphContext     *gc,              // graph context; used for schema lookups
						               // and attribute ID resolution

    StagedUpdatesCtx *staged_updates,  // accumulates all pending attribute
									   // and label changes until the
									   // transaction commits

    const Record *recs,                // array of records to process;
									   // each record identifies the concrete
									   // entity bound to each alias
    uint32_t n_recs,                   // number of records in 'recs'

    EntityUpdateDesc **descs,          // array of update descriptors;
									   // one per updated alias, each carrying
									   // the ordered list of update expressions
    uint32_t n_descs,                  // number of descriptors in 'descs'

    bool allow_null                    // when true, T_NULL is added to the
									   // accepted-type mask, permitting
									   // attribute deletion via SET n.v = NULL
) {
	// quick return if there are no records to act upon
	if (n_recs == 0) {
		return true ;
	}

	// validations
	ASSERT (gc             != NULL) ;
	ASSERT (recs           != NULL) ;
	ASSERT (descs          != NULL) ;
	ASSERT (staged_updates != NULL) ;
	ASSERT (n_descs > 0) ;

	// get the effects-buffer
	// record every update in the effects-buffer
	// used for replication and persistency
	EffectsBuffer *eb = QueryCtx_GetEffectsBuffer () ;
	ASSERT (eb != NULL) ;

	//--------------------------------------------------------------------------
	// create missing labels & attribute IDs
	//--------------------------------------------------------------------------

	// queries such as:
	// MATCH (n)
	// SET n:NewLabel
	//
	// and
	//
	// MATCH (n)
	// SET n.new_attr = 4
	//
	// can introduce graph schema changes

	_UpdateSchemas (gc, descs, n_descs) ;

	// if we're converting a SET clause, NULL is acceptable
	// as it indicates an attribute deletion
	SIType accepted_types = SI_VALID_PROPERTY_VALUE ;
	if (allow_null) {
		accepted_types |= T_NULL ;
	}

	EntityID *node_ids = NULL ;  // collected node ids

	// foreach update descriptor
	for (uint32_t i = 0 ; i < n_descs ; i++) {

		EntityUpdateDesc *desc = descs [i] ;

		bool labels_modified =
			(array_len (desc->add_labels)    > 0 ||
			 array_len (desc->remove_labels) > 0 ) ;

		if (labels_modified) {
			if (node_ids == NULL) {
				node_ids = array_new (EntityID, n_recs) ;
			}
		}

		// foreach record:
		// 1. evaluate update expressions
		// 2. stage label addition / removal
		for (uint32_t j = 0 ; j < n_recs ; j++) {
			Record rec = recs [j] ;

			//------------------------------------------------------------------
			// validate entity exists
			// is a graph entity
			// and supports the requested operation
			//------------------------------------------------------------------

			// get the type of the entity to update
			RecordEntryType entry_type = Record_GetType (rec, desc->record_idx) ;

			// a missing entity (e.g. from an OPTIONAL MATCH)
			// is silently skipped per Cypher semantics
			if (unlikely (entry_type == REC_TYPE_UNKNOWN)) {
				continue ;
			}

			// make sure we're updating a graph entity
			if (unlikely (entry_type != REC_TYPE_NODE &&
						  entry_type != REC_TYPE_EDGE)) {
				ErrorCtx_RaiseRuntimeException (
					"Update error: alias '%s' did not resolve to a graph entity",
					desc->alias) ;
				goto cleanup ;
			}

			// label(s) update can only be performed on nodes
			if (unlikely (labels_modified && entry_type != REC_TYPE_NODE)) {
				ErrorCtx_RaiseRuntimeException (
					"Label addition / removal can't be performed on an edge") ;
				goto cleanup ;
			}

			GraphEntity *entity = Record_GetGraphEntity (rec, desc->record_idx) ;

			// a concurrently deleted entity is silently skipped
			// deletion takes precedence over update
			if (unlikely (Graph_EntityIsDeleted (entity))) {
				continue ;
			}

			GraphEntityType entity_type = (entry_type == REC_TYPE_NODE) ?
				GETYPE_NODE :
				GETYPE_EDGE ;

			// evaluate update expressions
			if (!_UpdateEntity (gc, entity, entity_type, staged_updates, desc,
						rec, accepted_types, eb)) {
				goto cleanup ;
			}

			if (labels_modified) {
				array_append (node_ids, ENTITY_GET_ID (entity)) ;
			}
		} // end foreach record

		if (labels_modified) {
			uint32_t n_nodes = array_len (node_ids) ;
			if (unlikely (n_nodes == 0)) {
				continue ;
			}

			// label addition
			for (uint16_t j = 0 ; j < array_len (desc->add_labels) ; j++) {
				StagedUpdatesCtx_LabelNodes (staged_updates, node_ids,
						n_nodes, (char*) desc->add_labels [j]) ;
			}

			// label removal
			for (uint16_t j = 0 ; j < array_len (desc->remove_labels) ; j++) {
				StagedUpdatesCtx_UnLabelNodes (staged_updates, node_ids,
						n_nodes, (char*) desc->remove_labels [j]) ;
			}

			array_clear (node_ids) ;
		}
	} // end foreach update descriptor

cleanup:
	if (node_ids != NULL) {
		array_free (node_ids) ;
	}

	return !ErrorCtx_EncounteredError () ;
}

// make sure label matrices used in SET n:L and REMOVE n:M
// are of the correct dimensions NxN
void ensureMatrixDim
(
	GraphContext *gc,
	StagedUpdatesCtx *ctx
) {
	ASSERT (gc  != NULL) ;
	ASSERT (ctx != NULL) ;

	char         label[512] = {0}  ;
	Graph *g = GraphContext_GetGraph (gc) ;

	// set matrix sync policy to resize
	//MATRIX_POLICY policy = Graph_SetMatrixPolicy (g, SYNC_POLICY_RESIZE) ;

	//--------------------------------------------------------------------------
	// sync added label matrices
	//--------------------------------------------------------------------------

	uint8_t n_add = StagedUpdatesCtx_AddLabelCount (ctx) ;
	GrB_Vector *Vs = StagedUpdatesCtx_AddLabels (ctx) ;

	for (uint8_t i = 0 ; i < n_add ; i++) {
		GrB_Vector V = Vs [i] ;
		GrB_OK (GrB_get (V, label, GrB_NAME)) ;

		const Schema *s = GraphContext_GetSchema (gc, label, SCHEMA_NODE) ;
		ASSERT (s != NULL) ;

		// make sure label matrix is of the right dimensions
		Graph_GetLabelMatrix (g, Schema_GetID (s)) ;
	}

	//--------------------------------------------------------------------------
	// sync removed label matrices
	//--------------------------------------------------------------------------

	uint8_t n_rmv = StagedUpdatesCtx_RmvLabelCount (ctx) ;
	Vs = StagedUpdatesCtx_RmvLabels (ctx) ;

	for (uint8_t i = 0 ; i < n_rmv ; i++) {
		GrB_Vector V = Vs [i] ;
		GrB_OK (GrB_get (V, label, GrB_NAME)) ;

		const Schema *s = GraphContext_GetSchema (gc, label, SCHEMA_NODE) ;
		ASSERT (s != NULL) ;

		// make sure label matrix is of the right dimensions
		Graph_GetLabelMatrix (g, Schema_GetID (s)) ;
	}

	// sync node labels matrix
	if (n_rmv + n_add > 0) {
		Graph_GetNodeLabelMatrix (g) ;
	}

	// restore matrix sync policy
	//Graph_SetMatrixPolicy (g, policy) ;
}

void PendingUpdateCtx_Free
(
	PendingUpdateCtx *ctx
) {
	AttributeSet_Free (&ctx->attributes) ;
	rm_free (ctx) ;
}

