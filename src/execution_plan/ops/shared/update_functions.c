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
	struct hashmap *updates,
	EntityType type
) {
	ASSERT(gc      != NULL);
	ASSERT(updates != NULL);
	ASSERT(type    != ENTITY_UNKNOWN);

	uint update_count         = hashmap_count(updates);
	bool constraint_violation = false;

	// return early if no updates are enqueued
	if(update_count == 0) return;

	size_t i = 0;
	PendingUpdateCtx **entry;
	MATRIX_POLICY policy = Graph_GetMatrixPolicy(gc->g);
	Graph_SetMatrixPolicy(gc->g, SYNC_POLICY_NOP);

	while(hashmap_iter(updates, &i, (void**)&entry)) {
		PendingUpdateCtx *update = *entry;

		// if entity has been deleted, perform no updates
		if(GraphEntity_IsDeleted(update->ge)) continue;

		AttributeSet_PersistValues(update->attributes);
		
		// update the attributes on the graph entity
		UpdateEntityProperties(gc, update->ge, update->attributes,
				type == ENTITY_NODE ? GETYPE_NODE : GETYPE_EDGE, true);
		update->attributes = NULL;

		if(type == ENTITY_NODE) {
			UpdateNodeLabels(gc, (Node*)update->ge, update->add_labels,
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
}

// build pending updates in the 'updates' array to match all
// AST-level updates described in the context
// NULL values are allowed in SET clauses but not in MERGE clauses
void EvalEntityUpdates
(
	GraphContext *gc,
	struct hashmap *node_updates,
	struct hashmap *edge_updates,
	const Record r,
	const EntityUpdateEvalCtx *ctx,
	bool allow_null
) {
	Schema *s = NULL;

	//--------------------------------------------------------------------------
	// validate entity type
	//--------------------------------------------------------------------------

	// get the type of the entity to update
	// if the expected entity was not found, make no updates but do not error
	RecordEntryType t = Record_GetType(r, ctx->record_idx);
	if(unlikely(t == REC_TYPE_UNKNOWN)) {
		return;
	}

	// make sure we're updating either a node or an edge
	if(unlikely(t != REC_TYPE_NODE && t != REC_TYPE_EDGE)) {
		ErrorCtx_RaiseRuntimeException(
			"Update error: alias '%s' did not resolve to a graph entity",
			ctx->alias);
	}

	// label(s) update can only be performed on nodes
	if(unlikely((ctx->add_labels != NULL || ctx->remove_labels != NULL) &&
			t != REC_TYPE_NODE)) {
		ErrorCtx_RaiseRuntimeException(
				"Type mismatch: expected Node but was Relationship");
	}

	GraphEntity *entity = Record_GetGraphEntity(r, ctx->record_idx);

	// if the entity is marked as deleted, make no updates but do not error
	if(unlikely(Graph_EntityIsDeleted(entity))) {
		return;
	}

	struct hashmap *updates;
	GraphEntityType entity_type;
	if(t == REC_TYPE_NODE) {
		updates = node_updates;
		entity_type = GETYPE_NODE;
	} else {
		updates = edge_updates;
		entity_type = GETYPE_EDGE;
	}

	PendingUpdateCtx *update;
	PendingUpdateCtx **entry = (PendingUpdateCtx **)hashmap_get_with_hash(updates, NULL, ENTITY_GET_ID(entity));
	if(entry == NULL) {
		// create a new update context
		update = rm_malloc(sizeof(PendingUpdateCtx));
		update->ge            = entity;
		update->attributes    = AttributeSet_ShallowClone(*entity->attributes);
		update->add_labels    = NULL;
		update->remove_labels = NULL;
		// add update context to updates dictionary
		hashmap_set_with_hash(updates, &update, ENTITY_GET_ID(entity));
		entry = (PendingUpdateCtx **)hashmap_get_with_hash(updates, NULL, ENTITY_GET_ID(entity));
	} else {
		// update context already exists
		update = *entry;
	}

	if(array_len(ctx->add_labels) > 0 && update->add_labels == NULL) {
		update->add_labels = array_new(const char *, array_len(ctx->add_labels));
	}

	if(array_len(ctx->remove_labels) > 0 && update->remove_labels == NULL) {
		update->remove_labels = array_new(const char *, array_len(ctx->remove_labels));
	}
	
	array_union(update->add_labels, ctx->add_labels, strcmp);
	array_union(update->remove_labels, ctx->remove_labels, strcmp);

	AttributeSet *old_attrs = entity->attributes;
	entity->attributes = &update->attributes;
	if(t == REC_TYPE_NODE) {
		Record_AddNode(r, ctx->record_idx, *(Node *)entity);
	} else {
		Record_AddEdge(r, ctx->record_idx, *(Edge *)entity);
	}

	// if we're converting a SET clause, NULL is acceptable
	// as it indicates a deletion
	SIType accepted_properties = SI_VALID_PROPERTY_VALUE;
	if(allow_null) {
		accepted_properties |= T_NULL;
	}

	bool error = false;
	uint exp_count = array_len(ctx->properties);
	EffectsBuffer *eb = QueryCtx_GetEffectsBuffer();

	// evaluate each assigned expression
	// e.g. n.v = n.a + 2
	//
	// validate each new value type
	// e.g. invalid n.v = [1, {}]
	//
	// collect all updates into a single attribute-set
	//
	QueryCtx *query_ctx = QueryCtx_GetQueryCtx();
	for(uint i = 0; i < exp_count && !error; i++) {
		PropertySetCtx *property = ctx->properties + i;

		SIValue     v         = AR_EXP_Evaluate(property->exp, r);
		SIType      t         = SI_TYPE(v);
		UPDATE_MODE mode      = property->mode;
		const char* attribute = property->attribute;

		//----------------------------------------------------------------------
		// n.v = 2
		//----------------------------------------------------------------------

		if(attribute != NULL) {
			// a specific attribute is set, validate the value type
			if(!_ValidateAttrType(accepted_properties, v)) {
				error = true;
				SIValue_Free(v);
				Error_InvalidPropertyValue();
				break;
			}

			AttributeID attr_id = FindOrAddAttribute(gc, attribute, true);

			switch (AttributeSet_Set_Allow_Null(entity->attributes, attr_id, v))
			{
				case CT_DEL:
					// attribute removed
					EffectsBuffer_AddEntityRemoveAttributeEffect(eb, entity,
							attr_id, entity_type);
					continue;
				case CT_ADD:
					// attribute added
					EffectsBuffer_AddEntityAddAttributeEffect(eb, entity,
							attr_id, v, entity_type);
					break;
				case CT_UPDATE:
					// attribute update
					EffectsBuffer_AddEntityUpdateAttributeEffect(eb, entity,
							attr_id, v, entity_type);
					break;
				case CT_NONE:
					// no change
					break;
				default:
					assert("unknown change type value" && false);
					break;
			}
			SIValue_Free(v);
			continue;
		}

		//----------------------------------------------------------------------
		// n = {v:2}, n = m
		//----------------------------------------------------------------------

		if(!(t & (T_NODE | T_EDGE | T_MAP))) {
			error = true;
			SIValue_Free(v);
			Error_InvalidPropertyValue();
			break;
		}

		if(mode == UPDATE_REPLACE) {
			// if this update replaces all existing properties
			// enqueue a 'clear' update to do so
			EffectsBuffer_AddEntityRemoveAttributeEffect(eb, entity,
							ATTRIBUTE_ID_ALL, entity_type);
			AttributeSet_Free(entity->attributes);
		}

		//----------------------------------------------------------------------
		// n = {v:2}
		//----------------------------------------------------------------------

		if(t == T_MAP) {
			// value is of type map e.g. n.v = {a:1, b:2}
			// iterate over all map elements to build updates
			uint map_size = Map_KeyCount(v);
			for(uint j = 0; j < map_size; j ++) {
				SIValue key;
				SIValue value;

				Map_GetIdx(v, j, &key, &value);

				if(!_ValidateAttrType(accepted_properties, value)) {
					error = true;
					Error_InvalidPropertyValue();
					break;
				}

				AttributeID attr_id = FindOrAddAttribute(gc, key.stringval, true);
				// TODO: would have been nice we just sent n = {v:2}
				switch (AttributeSet_Set_Allow_Null(entity->attributes, attr_id, value))
				{
					case CT_DEL:
						// attribute removed
						EffectsBuffer_AddEntityRemoveAttributeEffect(eb, entity,
								attr_id, entity_type);
						break;
					case CT_ADD:
						// attribute added
						EffectsBuffer_AddEntityAddAttributeEffect(eb, entity,
								attr_id, value, entity_type);
						break;
					case CT_UPDATE:
						// attribute update
						EffectsBuffer_AddEntityUpdateAttributeEffect(eb, entity,
								attr_id, value, entity_type);
						break;
					case CT_NONE:
						// no change
						break;
					default:
						assert("unknown change type value" && false);
						break;
				}
			}

			// free map
			SIValue_Free(v);
			continue;
		}

		//----------------------------------------------------------------------
		// n = m
		//----------------------------------------------------------------------

		// value is a node or edge; perform attribute set reassignment
		ASSERT((t & (T_NODE | T_EDGE)));

		GraphEntity *ge = v.ptrval;

		// iterate over all entity properties to build updates
		const AttributeSet set = GraphEntity_GetAttributes(ge);

		for(uint j = 0; j < AttributeSet_Count(set); j++) {
			AttributeID attr_id;
			SIValue v = AttributeSet_GetIdx(set, j, &attr_id);

			// simple assignment, no need to validate value
			switch (AttributeSet_Set_Allow_Null(entity->attributes, attr_id, v))
			{
				case CT_ADD:
					// attribute added
					EffectsBuffer_AddEntityAddAttributeEffect(eb, entity,
							attr_id, v, entity_type);
					break;
				case CT_UPDATE:
					// attribute update
					EffectsBuffer_AddEntityUpdateAttributeEffect(eb, entity,
							attr_id, v, entity_type);
					break;
				default:
					break;
			}
		}
	} // for loop end

	// restore original attribute-set
	// changes should not be visible prior to the commit phase
	update->attributes = *entity->attributes;
	entity->attributes = old_attrs;
	if(t == REC_TYPE_NODE) {
		Record_AddNode(r, ctx->record_idx, *(Node *)entity);
	} else {
		Record_AddEdge(r, ctx->record_idx, *(Edge *)entity);
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
