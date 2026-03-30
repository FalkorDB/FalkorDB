/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// commit node updates
static void _CommitNodeUpdates
(
	GraphContext *gc,
	StagedUpdatesCtx *ctx
) {
	ASSERT (gc  != NULL) ;
	ASSERT (ctx != NULL) ;
	ASSERT (StagedUpdatesCtx_HasNodeUpdates (ctx)) ;

	bool enforce_constraints  = GraphContext_HasConstraints (gc) ;
	bool constraint_violation = false ;

	dictEntry *entry ;
	dict *updates = StagedUpdatesCtx_NodeUpdates (ctx) ;
	dictIterator *it = HashTableGetIterator (updates) ;

	MATRIX_POLICY policy = Graph_GetMatrixPolicy (gc->g) ;
	Graph_SetMatrixPolicy (gc->g, SYNC_POLICY_NOP) ;

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
				GETYPE_NODE, true) ;
		update->attributes = NULL ;

	}

	uint8_t n_add = StagedUpdatesCtx_AddLabelCount (ctx) ;
	uint8_t n_rmv = StagedUpdatesCtx_RmvLabelCount (ctx) ;
	if (n_add > 0 || n_rmv > 0) {
		GrB_Vector *add = StagedUpdatesCtx_AddLabels (ctx) ;
		GrB_Vector *rmv = StagedUpdatesCtx_RmvLabels (ctx) ;
		GraphHub_UpdateNodeLabels (gc, add, n_add, rmv, n_rmv, true) ;
	}

	//--------------------------------------------------------------------------
	// enforce constraints
	//--------------------------------------------------------------------------

	if (!enforce_constraints) {
		goto cleanup ;
	}

	uint graph_label_count = Graph_LabelTypeCount (gc->g) ;
	HashTableResetIterator (it) ;

	while ( constraint_violation == false &&
			(entry = HashTableNext (it)) != NULL) {
		PendingUpdateCtx *update = HashTableGetVal (entry) ;

		// if entity has been deleted, perform no updates
		if (GraphEntity_IsDeleted (update->ge)) {
			continue ;
		}

		LabelID labels [graph_label_count] ;
		uint label_count = Graph_GetNodeLabels (gc->g,
				(Node*)update->ge, labels, graph_label_count) ;

		for (uint i = 0; i < label_count; i ++) {
			Schema *s = GraphContext_GetSchemaByID (gc, labels[i],
					SCHEMA_NODE) ;
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

cleanup:
	Graph_SetMatrixPolicy (gc->g, policy) ;
	HashTableReleaseIterator (it) ;
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

	bool enforce_constraints  = GraphContext_HasConstraints (gc) ;
	bool constraint_violation = false ;

	dictEntry *entry ;
	dict *updates = StagedUpdatesCtx_EdgeUpdates (ctx) ;
	dictIterator *it = HashTableGetIterator (updates) ;

	MATRIX_POLICY policy = Graph_GetMatrixPolicy (gc->g) ;
	Graph_SetMatrixPolicy (gc->g, SYNC_POLICY_NOP) ;

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
				GETYPE_EDGE, true) ;
		update->attributes = NULL ;

		//----------------------------------------------------------------------
		// enforce constraints
		//----------------------------------------------------------------------

		if (!enforce_constraints) {
			continue ;	
		}

		RelationID rel_id = Edge_GetRelationID ((Edge*)update->ge) ;

		char *err_msg = NULL ;
		Schema *s = GraphContext_GetSchemaByID (gc, rel_id, SCHEMA_EDGE) ;
		if (!Schema_EnforceConstraints (s, update->ge, &err_msg)) {
			// constraint violation
			ASSERT (err_msg != NULL) ;
			constraint_violation = true ;
			ErrorCtx_SetError ("%s", err_msg) ;
			free (err_msg) ;
			break ;
		}
	}

	Graph_SetMatrixPolicy (gc->g, policy) ;
	HashTableReleaseIterator (it) ;
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

