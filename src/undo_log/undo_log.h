/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdint.h>
#include "../value.h"
#include "../schema/schema.h"
#include "../graph/graphcontext.h"
#include "../graph/entities/node.h"
#include "../graph/entities/edge.h"

// UndoLog
// matains a list of undo operation reverting all changes
// performed by a query: CREATE, UPDATE, DELETE
//
// upon failure for which ever reason we can apply the
// operations within the undo log to rollback the graph to its
// original state

// container for undo_list
typedef struct Opaque_UndoLog *UndoLog;

// create a new undo-log
UndoLog UndoLog_New(void);

// returns number of entries in log
uint UndoLog_Length
(
	const UndoLog log  // log to query
);

//------------------------------------------------------------------------------
// UndoLog add operations
//------------------------------------------------------------------------------

// undo node creation
void UndoLog_CreateNode
(
	UndoLog log,   // undo log
	Node *node     // node created
);

// undo edge creation
void UndoLog_CreateEdge
(
	UndoLog log,   // undo log
	Edge *edge     // edge created
);

// undo node deletion
void UndoLog_DeleteNode
(
	UndoLog log,      // undo log
	Node *node,       // node deleted
	LabelID *labels,  // labels attached to deleted entity
	uint label_count  // number of labels attached to deleted entity
);

// undo edge deletion
void UndoLog_DeleteEdge
(
	UndoLog log,   // undo log
	Edge *edge     // edge deleted
);

// undo node update
void UndoLog_UpdateNode
(
	UndoLog log,                 // undo log
	Node *n,                     // updated node
	AttributeSet set             // old attribute set
);

// undo entity update
void UndoLog_UpdateEdge
(
	UndoLog log,                 // undo log
	Edge *ge,                    // updated edge
	AttributeSet set             // old attribute set
);

// undo node add label
void UndoLog_AddLabels
(
	UndoLog log,                 // undo log
	Node *node,                  // updated node
	LabelID *label_ids,          // added labels
	size_t labels_count          // number of removed labels
);

// undo node remove label
void UndoLog_RemoveLabels
(
	UndoLog log,                 // undo log
	Node *node,                  // updated node
	LabelID *label_ids,          // removed labels
	size_t labels_count          // number of removed labels
);

// undo schema addition
void UndoLog_AddSchema
(
	UndoLog log,                 // undo log
	int schema_id,               // id of the schema
	SchemaType t                 // type of the schema
);

// undo attribute addition
void UndoLog_AddAttribute
(
	UndoLog log,                 // undo log
	AttributeID attribute_id     // id of the attribute
);

// undo index creation
void UndoLog_CreateIndex
(
	UndoLog log,                 // undo log
	SchemaType st,               // schema type
	const char *label,           // label / relationship
	const char *field,           // attribute
	IndexFieldType t             // type of index
);

// rollback all modifications tracked by this undo log
void UndoLog_Rollback
(
	UndoLog log,
	GraphContext *gc
);

// free UndoLog
void UndoLog_Free
(
	UndoLog log
);
