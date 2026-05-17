/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "update_functions.h"
#include "../../../errors/errors.h"
#include "../../../graph/graph_hub.h"
#include "../../../graph/graphcontext.h"

static void _EnforceConstraints
(
	GraphContext *gc,
	Graph *g,
	dict *entities,
	GraphEntityType t
) {
	ASSERT (g        != NULL) ;
	ASSERT (gc       != NULL) ;
	ASSERT (entities != NULL) ;

	dictEntry *entry ;
	dictIterator *it = HashTableGetIterator (entities) ;

	//--------------------------------------------------------------------------
	// enforce constraints
	//--------------------------------------------------------------------------

	Schema *s                    = NULL ;
	char   *err_msg              = NULL ;
	bool    constraint_violation = false ;

	//--------------------------------------------------------------------------
	// enforce edge constraints
	//--------------------------------------------------------------------------

	if (t == GETYPE_EDGE) {
		while (constraint_violation == false &&
				(entry = HashTableNext (it)) != NULL) {

			PendingUpdateCtx *update = HashTableGetVal (entry) ;
			Edge *e = (Edge*)update->ge ;

			RelationID rel_id = Edge_GetRelationID (e) ;
			s = GraphContext_GetSchemaByID (gc, rel_id, SCHEMA_EDGE) ;
			ASSERT (s != NULL) ;

			if (!Schema_EnforceConstraints (s, (GraphEntity*) e, &err_msg)) {
				// constraint violation
				ASSERT (err_msg != NULL) ;
				constraint_violation = true ;
				ErrorCtx_SetError ("%s", err_msg) ;
				free (err_msg) ;
				break ;
			}
		}

		HashTableReleaseIterator (it) ;
		return ;
	}

	//--------------------------------------------------------------------------
	// enforce node constraints
	//--------------------------------------------------------------------------

	uint graph_label_count = Graph_LabelTypeCount (g) ;

	while (constraint_violation == false &&
           (entry = HashTableNext (it)) != NULL) {

		PendingUpdateCtx *update = HashTableGetVal (entry) ;
		Node *n = (Node*) update->ge ;

		LabelID labels [graph_label_count] ;
		uint lbl_count = Graph_GetNodeLabels (g, n, labels, graph_label_count) ;

		for (uint i = 0; i < lbl_count; i ++) {
			s = GraphContext_GetSchemaByID (gc, labels [i], SCHEMA_NODE) ;
			ASSERT (s != NULL) ;

			// TODO: a bit wasteful need to target relevant constraints only
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

	HashTableReleaseIterator (it) ;
}

// commit node updates
static void _CommitNodeUpdates
(
	GraphContext *gc,
	StagedUpdatesCtx *ctx
) {
	ASSERT (gc  != NULL) ;
	ASSERT (ctx != NULL) ;
	ASSERT (StagedUpdatesCtx_HasNodeUpdates (ctx)) ;

	Graph *g = GraphContext_GetGraph (gc) ;
	MATRIX_POLICY policy = Graph_GetMatrixPolicy (g) ;
	Graph_SetMatrixPolicy (g, SYNC_POLICY_NOP) ;

	dictEntry *entry ;
	dict *updates = StagedUpdatesCtx_NodeUpdates (ctx) ;
	dictIterator *it = HashTableGetIterator (updates) ;

	while ((entry = HashTableNext(it)) != NULL) {
		PendingUpdateCtx *update = HashTableGetVal (entry) ;

		// update the attributes on the graph entity
		GraphHub_UpdateEntityProperties (gc, update->ge, update->attributes,
				GETYPE_NODE, true) ;

		update->attributes = NULL ;
	}

	HashTableReleaseIterator (it) ;

	uint8_t n_add = StagedUpdatesCtx_AddLabelCount (ctx) ;
	uint8_t n_rmv = StagedUpdatesCtx_RmvLabelCount (ctx) ;

	if (n_add > 0 || n_rmv > 0) {
		GrB_Vector *add = StagedUpdatesCtx_AddLabels (ctx) ;
		GrB_Vector *rmv = StagedUpdatesCtx_RmvLabels (ctx) ;
		ASSERT (n_add > 0 && add != NULL || n_add == 0 && add == NULL) ;
		ASSERT (n_rmv > 0 && rmv != NULL || n_rmv == 0 && rmv == NULL) ;
		ASSERT (add != NULL || rmv != NULL) ;
		GraphHub_UpdateNodeLabels (gc, add, n_add, rmv, n_rmv, true) ;
	}

	if (GraphContext_HasConstraints (gc)) {
		_EnforceConstraints (gc, g, updates, GETYPE_NODE) ;
	}

	Graph_SetMatrixPolicy (g, policy) ;
}

// commit edge updates
static void _CommitEdgeUpdates
(
	GraphContext *gc,
	StagedUpdatesCtx *ctx
) {
	ASSERT (gc  != NULL) ;
	ASSERT (ctx != NULL) ;
	ASSERT (StagedUpdatesCtx_HasEdgeUpdates (ctx)) ;

	Graph *g = GraphContext_GetGraph (gc) ;
	MATRIX_POLICY policy = Graph_GetMatrixPolicy (g) ;
	Graph_SetMatrixPolicy (g, SYNC_POLICY_NOP) ;

	dictEntry *entry ;
	dict *updates = StagedUpdatesCtx_EdgeUpdates (ctx) ;
	dictIterator *it = HashTableGetIterator (updates) ;

	while ((entry = HashTableNext(it)) != NULL) {
		PendingUpdateCtx *update = HashTableGetVal (entry) ;

		// update the attributes on the graph entity
		GraphHub_UpdateEntityProperties (gc, update->ge, update->attributes,
				GETYPE_EDGE, true) ;

		update->attributes = NULL ;
	}

	HashTableReleaseIterator (it) ;

	if (GraphContext_HasConstraints (gc)) {
		_EnforceConstraints (gc, g, updates, GETYPE_EDGE) ;
	}

	Graph_SetMatrixPolicy (g, policy) ;
}

// commit all updates described in the array of pending updates
void CommitUpdates
(
	GraphContext *gc,
	StagedUpdatesCtx *updates
) {
	ASSERT (gc      != NULL) ;
	ASSERT (updates != NULL) ;

	// return early if no updates are enqueued
	if (!StagedUpdatesCtx_HasNodeUpdates (updates) &&
		!StagedUpdatesCtx_HasEdgeUpdates (updates)) {
		return ;
	}

	// commit node updates
	if (StagedUpdatesCtx_HasNodeUpdates (updates)) {
		_CommitNodeUpdates (gc, updates) ;
	}

	// return if we've encountered an error
	if (ErrorCtx_EncounteredError ()) {
		return ;
	}

	// commit edge updates
	if (StagedUpdatesCtx_HasEdgeUpdates (updates)) {
		_CommitEdgeUpdates (gc, updates) ;
	}
}

